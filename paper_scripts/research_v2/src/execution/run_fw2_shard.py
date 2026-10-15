#!/usr/bin/env python3
"""Run a bounded, resumable shard of frozen FW2 jobs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from research_v2.src.execution.run_fw2_job import FORMAL_ROOT, verify_completed_directory


V2_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = V2_ROOT / "data_registry" / "fw2_inner_benchmark_job_manifest.csv"
SHARD_LOG_DIR = V2_ROOT / "logs" / "fw2_shards"
PRIVATE_CAPTURE_DIR = V2_ROOT / "outputs" / "private" / "fw2" / "shard_subprocess_logs"


@contextmanager
def keep_system_awake():
    """Prevent Windows system sleep only while a formal shard is active."""
    active = False
    if sys.platform == "win32":
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000001
        result = ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
        if result == 0:
            raise OSError("SetThreadExecutionState failed")
        active = True
    try:
        yield active
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def select_manifest_jobs(rows: list[dict[str, str]], offset: int, count: int) -> list[dict[str, str]]:
    if offset < 0 or count <= 0:
        raise ValueError("offset must be non-negative and count must be positive")
    return rows[offset : offset + count]


def run_subprocess(job_id: str, threads: int, timeout_seconds: int) -> dict:
    command = [
        sys.executable,
        "-m",
        "research_v2.src.execution.run_fw2_job",
        "--job-id",
        job_id,
        "--mode",
        "formal",
        "--threads",
        str(threads),
    ]
    started = time.perf_counter()
    PRIVATE_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    timed_out = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=PRIVATE_CAPTURE_DIR,
            prefix=f".{job_id}-", suffix=".stdout.tmp", delete=False,
        ) as stdout_handle, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=PRIVATE_CAPTURE_DIR,
            prefix=f".{job_id}-", suffix=".stderr.tmp", delete=False,
        ) as stderr_handle:
            stdout_path = Path(stdout_handle.name)
            stderr_path = Path(stderr_handle.name)
            try:
                completed = subprocess.run(
                    command,
                    cwd=V2_ROOT.parent,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                    close_fds=True,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    finally:
        if stdout_path is not None:
            stdout_path.unlink(missing_ok=True)
        if stderr_path is not None:
            stderr_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    if timed_out:
        if verify_completed_directory(FORMAL_ROOT / job_id):
            return {
                "job_id": job_id,
                "status": "PASS",
                "action": "RECOVER_VERIFIED_AFTER_TIMEOUT",
                "return_code": 0,
                "wall_seconds": elapsed,
                "error_tail": "",
            }
        return {
            "job_id": job_id,
            "status": "FAIL",
            "action": "TIMEOUT",
            "return_code": -1,
            "wall_seconds": elapsed,
            "error_tail": "job timeout",
        }
    if completed is None:
        raise RuntimeError("Subprocess ended without a result or timeout")
    status = "PASS" if completed.returncode == 0 else "FAIL"
    action = "SKIP" if "SKIP verified existing" in stdout_text else "RUN"
    return {
        "job_id": job_id,
        "status": status,
        "action": action,
        "return_code": completed.returncode,
        "wall_seconds": elapsed,
        "error_tail": "" if status == "PASS" else stderr_text[-1000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-job", type=int, default=2)
    parser.add_argument("--job-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not args.shard_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("shard-id must contain only letters, numbers, hyphens, or underscores")
    if args.workers < 1 or args.threads_per_job < 1:
        raise ValueError("workers and threads-per-job must be positive")

    SHARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    output = SHARD_LOG_DIR / f"{args.shard_id}.json"
    if output.exists():
        raise RuntimeError(f"Shard log already exists and will not be overwritten: {output}")

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_manifest_jobs(rows, args.offset, args.count)
    if not selected:
        raise ValueError("Shard selection is empty")

    started = time.perf_counter()
    results = []
    with keep_system_awake() as sleep_prevention_active:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    run_subprocess,
                    row["job_id"],
                    args.threads_per_job,
                    args.job_timeout_seconds,
                ): row["job_id"]
                for row in selected
            }
            for future in as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(
                        {
                            "job_id": future_map[future],
                            "status": "FAIL",
                            "action": "RUNNER_EXCEPTION",
                            "return_code": -1,
                            "wall_seconds": 0.0,
                            "error_tail": str(error)[-1000:],
                        }
                    )
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item["job_id"])
    failures = [item for item in results if item["status"] != "PASS"]
    payload = {
        "shard_id": args.shard_id,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "selection": {
            "offset": args.offset,
            "count_requested": args.count,
            "count_selected": len(selected),
            "first_job_id": selected[0]["job_id"],
            "last_job_id": selected[-1]["job_id"],
        },
        "execution": {
            "workers": args.workers,
            "threads_per_job": args.threads_per_job,
            "system_sleep_prevention_active_during_shard": sleep_prevention_active,
            "wall_seconds": elapsed,
            "run_jobs": sum(item["action"] == "RUN" for item in results),
            "skipped_jobs": sum(item["action"] == "SKIP" for item in results),
            "recovered_after_timeout_jobs": sum(
                item["action"] == "RECOVER_VERIFIED_AFTER_TIMEOUT" for item in results
            ),
            "failed_jobs": len(failures),
        },
        "results": results,
        "performance_reporting": "No performance metrics are read or recorded by the shard runner.",
        "privacy": "No participant identifiers, rows, predictions, representations, or performance metrics are present.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{payload['status']}: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
