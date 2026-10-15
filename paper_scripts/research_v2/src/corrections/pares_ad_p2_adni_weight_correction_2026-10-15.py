"""Correction of the ADNI locally refit sensitivity in P2 (2026-10-15).

Protocol Addendum B (pares_ad_p2_ensemble_baseline_2026-10-14.md) specifies
configuration-equal training weights taken from review_weight. The original
script (src/analysis/pares_ad_p2_ensemble_baseline_2026-10-14.py, crossfit_scores)
used review_weight for NACC but row-equal weights (np.ones) for ADNI.

This script reruns ONLY the ADNI local-refit block with the protocol weights,
keeping everything else identical: rows and row order, participant folds and
seed (20261014), features, logistic settings, budgets, bootstrap seed
(20261070) and B = 2000. It first re-executes the original row-equal version
as a parity check against the saved p2_results.csv, then the corrected version.
The original script and its outputs are not modified. Aggregate outputs only.
"""
import argparse, datetime, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AN = HERE.parent / 'analysis'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, AN / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


p2 = load('p2', 'pares_ad_p2_ensemble_baseline_2026-10-14.py')
p1 = p2.p1


def crossfit(df, weights, folds=5, seed=20261014):
    pk = df.participant_key.to_numpy()
    uniq = pd.unique(pk)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, folds, len(uniq))))
    fold = np.array([fold_of[k] for k in pk])
    w_all = weights / weights.mean()
    y_all = ((df.model_probability_AD.to_numpy(float) >= .5).astype(int) != df.AD_label.to_numpy(int)).astype(float)
    designs = {'Cx': p1.design(df), 'CEx': p2.design_ce(df, 'participant_key', True),
               'Ax': p1.design(df, p1.extras(df, 'A')), 'Px': p1.design(df, p1.extras(df, 'P'))}
    out = {k: np.zeros(len(df)) for k in designs}
    for f in range(folds):
        tr, te = fold != f, fold == f
        for k, X in designs.items():
            h = p1.fit_head(X[tr], y_all[tr], w_all[tr] / w_all[tr].mean())
            out[k][te] = p1.predict_head(h, X[te])
    return out


def auroc(s, t):
    r = pd.Series(s).rank().to_numpy()
    n1 = t.sum(); n0 = len(t) - n1
    return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = p1.load_all(root)
    df = data['adni']
    coh = p1.prepare_cohort(df, 'adni')
    e = coh.e.astype(int)
    lm = ['Cx', 'CEx', 'Ax', 'Px']
    lpairs = [('CEx', 'Cx'), ('Px', 'Cx'), ('Ax', 'Cx'), ('Px', 'CEx'), ('Ax', 'CEx')]
    rows, info = [], {}
    for label, w in [('row_equal_original', np.ones(len(df))),
                     ('config_equal_protocol', df.review_weight.to_numpy(float))]:
        cs = crossfit(df, w)
        coh.set_scores(cs)
        info[label] = {('auroc_error_' + k): auroc(v, e) for k, v in cs.items()}
        pt, ptot, bo = p1.run_boot(coh, lm, 'participant', a.B, 20261070, None, True)
        r = p1.summarize('adni', 'local_refit', lm, pt, ptot, bo, lpairs, 'participant', a.B)
        for x in r:
            x['weighting'] = label
        rows += r
        print(label, 'done', round(time.time() - t0), 's', flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'adni_local_refit_weight_correction.csv', index=False, float_format='%.6g')
    # parity of the row-equal rerun against the saved P2 output
    old = pd.read_csv(root / 'research_v2/outputs/private/pares_ad_p2_ensemble_baseline_2026-10-14/p2_results.csv')
    old = old[(old.cohort == 'adni') & (old.scenario == 'local_refit')]
    new = res[res.weighting == 'row_equal_original']
    keys = ['kind', 'method', 'contrast', 'budget']
    m = old.merge(new, on=[k for k in keys if k in old.columns and k in new.columns], suffixes=('_old', '_new'))
    cols = [c for c in ['yield_pct', 'estimate_pp', 'ci_low_pp', 'ci_high_pp'] if c + '_old' in m]
    maxdiff = float(np.nanmax(np.abs(np.concatenate([(m[c + '_old'] - m[c + '_new']).to_numpy(float) for c in cols]))))
    receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   purpose='ADNI local-refit training weights corrected to protocol Addendum B (config-equal from review_weight)',
                   script_sha256=p1.sha(Path(__file__)),
                   original_script_sha256=p1.sha(AN / 'pares_ad_p2_ensemble_baseline_2026-10-14.py'),
                   B=a.B, parity_rows_compared=int(len(m)), parity_max_abs_diff=maxdiff,
                   auroc=info, runtime_seconds=round(time.time() - t0))
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main()
