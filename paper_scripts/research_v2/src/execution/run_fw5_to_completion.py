#!/usr/bin/env python3
"""Run every frozen FW5 outer-refit job with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from research_v2.src.execution.fw5_common import (
    DEFINITION_PATH, MANIFEST_PATH, OUTPUT_ROOT, V2_ROOT, load_csv_rows, sha256_file,
    verify_completed_directory, verify_frozen_sources,
)
from research_v2.src.execution.run_fw2_shard import keep_system_awake


SESSION_ROOT = V2_ROOT / "logs" / "fw5_completion_sessions"


def run_one(job_id: str, timeout_seconds: int) -> dict[str, object]:
    command = [sys.executable, "-m", "research_v2.src.execution.run_fw5_job", "--job-id", job_id, "--device", "cpu", "--threads", "1"]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=V2_ROOT.parent, env=environment, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        return {"job_id": job_id, "returncode": result.returncode, "timed_out": False, "elapsed_seconds": time.perf_counter() - started, "stderr_tail": result.stderr[-2000:]}
    except subprocess.TimeoutExpired as error:
        return {"job_id": job_id, "returncode": None, "timed_out": True, "elapsed_seconds": time.perf_counter() - started, "stderr_tail": str(error.stderr or "")[-2000:]}


def write_status(path, *, session_id: str, status: str, workers: int, completed: set[str], failures: list[dict], started: float, durations: list[float]) -> None:
    payload = {
        "session_id": session_id,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "worker_threads": 1,
        "verified_jobs": len(completed),
        "remaining_jobs": 290 - len(completed),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_job_seconds_median": float(np.median(durations)) if durations else None,
        "technical_failures": failures,
        "checks": {"frozen_selection_recomputed": False, "outer_test_predictions_generated": len(completed) > 0},
        "source_sha256": {"manifest": sha256_file(MANIFEST_PATH), "definition": sha256_file(DEFINITION_PATH)},
        "privacy": "Counts, timing, status, and hashes only; no outer metrics or selected identifiers.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--job-timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.job_timeout_seconds < 300:
        raise ValueError("Invalid FW5 worker or timeout")
    verify_frozen_sources()
    rows = load_csv_rows(MANIFEST_PATH)
    if len(rows) != 290:
        raise RuntimeError("FW5 manifest requires 290 jobs")
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    progress = SESSION_ROOT / f"{args.session_id}.progress.json"
    summary = SESSION_ROOT / f"{args.session_id}.summary.json"
    if summary.exists():
        raise RuntimeError("FW5 session is already finalized")
    completed = {row["fw5_job_id"] for row in rows if verify_completed_directory(OUTPUT_ROOT / row["fw5_job_id"], row["fw5_job_id"])}
    remaining = [row["fw5_job_id"] for row in rows if row["fw5_job_id"] not in completed]
    failures: list[dict] = []
    durations: list[float] = []
    started = time.perf_counter()
    with keep_system_awake() as active:
        if sys.platform == "win32" and not active:
            raise RuntimeError("Windows sleep prevention did not activate")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, job_id, args.job_timeout_seconds): job_id for job_id in remaining}
            for future in as_completed(futures):
                result = future.result()
                job_id = str(result["job_id"])
                valid = verify_completed_directory(OUTPUT_ROOT / job_id, job_id)
                if valid and (result["returncode"] == 0 or result["timed_out"]):
                    completed.add(job_id)
                    durations.append(float(result["elapsed_seconds"]))
                else:
                    failures.append(result)
                write_status(progress, session_id=args.session_id, status="FAIL" if failures else "RUNNING", workers=args.workers, completed=completed, failures=failures, started=started, durations=durations)
                if failures:
                    for pending in futures:
                        pending.cancel()
                    break
    if failures or len(completed) != 290:
        raise RuntimeError(f"FW5 completion failed: completed={len(completed)}, failures={len(failures)}")
    write_status(summary, session_id=args.session_id, status="PASS", workers=args.workers, completed=completed, failures=[], started=started, durations=durations)
    if progress.exists():
        progress.unlink()
    print(f"PASS stop_reason=FW5_OUTER_REFITS_COMPLETE verified=290/290 elapsed_seconds={time.perf_counter()-started:.1f}")
    print(f"PASS: {summary}")


if __name__ == "__main__":
    main()
