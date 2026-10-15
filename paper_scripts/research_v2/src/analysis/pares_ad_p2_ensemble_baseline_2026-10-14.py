"""PARES-AD P2 supplement (post hoc; protocol: protocols/pares_ad_p2_ensemble_baseline_2026-10-14.md).

Cross-classifier disagreement (committee) baseline for fixed-budget review.
Reuses the P1 engine. numpy + pandas only. Aggregate outputs only.
"""
import argparse, datetime, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('p1', HERE / 'pares_ad_p1_supplement_2026-10-14.py')
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)


def committee_features(df, key):
    p = df.model_probability_AD.to_numpy(float)
    g = pd.Series(p).groupby(df[key].to_numpy())
    sd = g.transform(lambda s: s.std(ddof=0)).to_numpy(float)
    f = pd.Series((p >= .5).astype(float)).groupby(df[key].to_numpy()).transform('mean').to_numpy(float)
    return sd, 4 * f * (1 - f)


def rank_by_participant(df, key, v):
    t = pd.DataFrame({'k': df[key].to_numpy(), 'v': v}).drop_duplicates('k').set_index('k')
    r = t['v'].rank(pct=True)
    return df[key].map(r).to_numpy(float)


def design_ce(df, key, use_vote=True, rank=False):
    p = df.model_probability_AD.to_numpy(float)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.abs(np.log(pc / (1 - pc)))
    sd, vote = committee_features(df, key)
    if rank:
        sd, vote = rank_by_participant(df, key, sd), rank_by_participant(df, key, vote)
    cols = [z, z * z, (p >= .5).astype(float), sd]
    if use_vote:
        cols.append(vote)
    return np.column_stack(cols)


def crossfit_scores(df, name, folds=5, seed=20261014):
    """Cross-fitted local heads for C, CE, A, P (sensitivity; uses labels of the cohort)."""
    pk = df.participant_key.to_numpy()
    uniq = pd.unique(pk)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, folds, len(uniq))))
    fold = np.array([fold_of[k] for k in pk])
    if 'review_weight' in df and name != 'adni':
        w_all = df.review_weight.to_numpy(float)
    else:
        w_all = np.ones(len(df))
    w_all = w_all / w_all.mean()
    y_all = (df.model_probability_AD.to_numpy(float) >= .5).astype(int) != df.AD_label.to_numpy(int)
    y_all = y_all.astype(float)
    designs = {'Cx': p1.design(df), 'CEx': design_ce(df, 'participant_key', True),
               'Ax': p1.design(df, p1.extras(df, 'A')), 'Px': p1.design(df, p1.extras(df, 'P'))}
    out = {k: np.zeros(len(df)) for k in designs}
    for f in range(folds):
        tr, te = fold != f, fold == f
        for k, X in designs.items():
            h = p1.fit_head(X[tr], y_all[tr], w_all[tr] / w_all[tr].mean())
            out[k][te] = p1.predict_head(h, X[te])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--Bc', type=int, default=1000)
    ap.add_argument('--protocol', default='')
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = p1.load_all(root)
    base = ['C', 'A', 'P']
    rows, info = [], {}
    for k, name in enumerate(['adni', 'uid', 'unique_date_description']):
        src = data['src_full'] if name == 'adni' else data['src_masked']
        w, y = p1.src_weights(src), p1.src_target(src)
        heads = {'CE': p1.fit_head(design_ce(src, '_RID', True), y, w),
                 'CEs': p1.fit_head(design_ce(src, '_RID', False), y, w),
                 'CEr': p1.fit_head(design_ce(src, '_RID', True, True), y, w)}
        df = data[name]
        scores = {m: df[p1.METHOD_ALIAS[m]].to_numpy(float) for m in base}
        scores['CE'] = p1.predict_head(heads['CE'], design_ce(df, 'participant_key', True))
        scores['CEs'] = p1.predict_head(heads['CEs'], design_ce(df, 'participant_key', False))
        scores['CEr'] = p1.predict_head(heads['CEr'], design_ce(df, 'participant_key', True, True))
        coh = p1.prepare_cohort(df, name)
        coh.set_scores(scores)
        sd, vote = committee_features(df, 'participant_key')
        e = coh.e
        # descriptive: does committee spread rank errors at all (row-level AUROC via rank statistic)
        from math import isfinite
        def auroc(s, t):
            r = pd.Series(s).rank().to_numpy()
            n1 = t.sum(); n0 = len(t) - n1
            return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        info[name] = dict(rows=int(len(df)), participants=int(df.participant_key.nunique()),
                          mean_E_sd=float(sd.mean()), source_mean_E_sd=float(committee_features(src, '_RID')[0].mean()),
                          beta_CE=[float(x) for x in heads['CE']['beta']],
                          auroc_error_C=auroc(df.confidence_only.to_numpy(float), e.astype(int)),
                          auroc_error_CE=auroc(scores['CE'], e.astype(int)),
                          auroc_error_CEs=auroc(scores['CEs'], e.astype(int)),
                          auroc_error_CEr=auroc(scores['CEr'], e.astype(int)),
                          auroc_error_A=auroc(scores['A'], e.astype(int)),
                          auroc_error_P=auroc(scores['P'], e.astype(int)))
        methods = ['C', 'CE', 'CEs', 'CEr', 'A', 'P']
        pairs = [('CE', 'C'), ('CEs', 'C'), ('CEr', 'C'), ('A', 'CE'), ('P', 'CE'), ('A', 'CEs'), ('P', 'CEs'), ('A', 'CEr'), ('P', 'CEr'), ('A', 'C'), ('P', 'C')]
        pt, ptot, bo = p1.run_boot(coh, methods, 'participant', args.B, 20261060 + k, None, True)
        rows += p1.summarize(name, 'committee_baseline', methods, pt, ptot, bo, pairs, 'participant', args.B)
        if name != 'adni':
            pt, ptot, bo = p1.run_boot(coh, methods, 'center', args.Bc, 20261160 + k, None, True)
            rows += p1.summarize(name, 'committee_baseline', methods, pt, ptot, bo, pairs, 'center', args.Bc)
        # cross-fitted local-refit sensitivity (Addendum B)
        cs = crossfit_scores(df, name)
        coh.set_scores(cs)
        lm = ['Cx', 'CEx', 'Ax', 'Px']
        lpairs = [('CEx', 'Cx'), ('Px', 'Cx'), ('Ax', 'Cx'), ('Px', 'CEx'), ('Ax', 'CEx')]
        for kk, sc in cs.items():
            info[name]['auroc_error_' + kk] = auroc(sc, e.astype(int))
        pt, ptot, bo = p1.run_boot(coh, lm, 'participant', args.B, 20261070 + k, None, True)
        rows += p1.summarize(name, 'local_refit', lm, pt, ptot, bo, lpairs, 'participant', args.B)
        if name != 'adni':
            pt, ptot, bo = p1.run_boot(coh, lm, 'center', args.Bc, 20261170 + k, None, True)
            rows += p1.summarize(name, 'local_refit', lm, pt, ptot, bo, lpairs, 'center', args.Bc)
        print(name, 'done', round(time.time() - t0), 's', json.dumps(info[name]), flush=True)
    pd.DataFrame(rows).to_csv(out / 'p2_results.csv', index=False, float_format='%.6g')
    receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='posthoc_exploratory',
                   script_sha256=p1.sha(Path(__file__)), protocol_sha256=p1.sha(Path(args.protocol)) if args.protocol else None,
                   B_participant=args.B, B_center=args.Bc, cohorts=info, runtime_seconds=round(time.time() - t0))
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print('DONE', receipt['runtime_seconds'], 's')


if __name__ == '__main__':
    main()
