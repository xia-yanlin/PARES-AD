"""Parity check of this library against the saved scores and results of the original analysis.

Requires the restricted ADNI/NACC score files of the original project; it cannot run from this package alone.

Usage: python tests/parity_with_saved_scores.py --root <original project root>

1. Refits the source-fitted review heads C, A, K and P with this library and compares them with the saved scores
   in ADNI and both NACC cohorts (tolerance 5e-3; the original analysis used L-BFGS, this library a Newton solver).
2. Reruns the ADNI fixed-budget bootstrap on the saved scores (2,000 participant-cluster resamples, seed 20261018)
   and compares the A-minus-C and P-minus-C contrasts with the original batch (Supplementary Data D8).
3. Recomputes the committee score CE and the locally refit scores Cx, CEx, Ax and Px in all three cohorts and
   compares their review yields at the 5%, 10% and 20% budgets with Table 5 / Supplementary Data D22.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pares_ad import (fit_head, predict_head, design, extras, source_weights, source_target, Cohort, run_boot, summarize,
                      committee_design, crossfit_scores)

ALIAS = {'C': 'confidence_only', 'A': 'confidence_plus_conflict', 'K': 'confidence_plus_control', 'P': 'confidence_plus_probability'}
COHORTS = ['adni', 'uid', 'unique_date_description']
BUDGETS = [.05, .10, .20]


def load(priv):
    tr = priv / 'pares_ad_comparator_transport_2026-10-08'
    dt = dict(participant_key=str, row_key=str, center=str, _RID=str)
    cohorts = {'adni': pd.read_csv(tr / 'adni_scores_private.csv', dtype=dt)}
    for name in COHORTS[1:]:
        s = pd.read_csv(tr / f'{name}_scores_private.csv', dtype=dt)
        y = pd.read_csv(priv / f'nacc_external_validation_2026-10-07/frozen_cohorts/{name}/outcomes.csv',
                        dtype=dict(participant_key=str, row_key=str))
        cohorts[name] = s.merge(y, on=['row_key', 'participant_key'], validate='many_to_one')
    full = pd.read_csv(tr / 'source_full_private.csv', dtype=dict(_RID=str))
    masked = pd.read_csv(tr / 'source_masked_private.csv', dtype=dict(_RID=str))
    return cohorts, {'adni': full, 'uid': masked, 'unique_date_description': masked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--B', type=int, default=2000)
    args = ap.parse_args()
    priv = Path(args.root) / 'research_v2/outputs/private'
    cohorts, src = load(priv)
    ok = True

    # 1. source-fitted heads against saved scores
    for name, df in cohorts.items():
        s = src[name]
        w, y = source_weights(s), source_target(s)
        gap = {}
        for k in 'CAKP':
            sc = predict_head(fit_head(design(s, extras(s, k)), y, w), design(df, extras(df, k)))
            gap[k] = float(np.max(np.abs(sc - df[ALIAS[k]].to_numpy(float))))
        print(f'[1] {name}: max |refit - saved| by score', {k: f'{v:.2e}' for k, v in gap.items()})
        ok &= max(gap.values()) < 5e-3

    # 2. ADNI bootstrap contrasts on saved scores
    df = cohorts['adni']
    coh = Cohort('adni', df, ['configuration_id'])
    coh.set_scores({k: df[ALIAS[k]].to_numpy(float) for k in 'CAKP'})
    pt, ptot, bo = run_boot(coh, list('CAKP'), 'participant', args.B, 20261018, None, True)
    rows = pd.DataFrame(summarize('adni', 'native', list('CAKP'), pt, ptot, bo, [('A', 'C'), ('P', 'C')], 'participant', args.B))
    ref = pd.read_csv(priv / 'pares_ad_p1_supplement_2026-10-14/s0_native/native_and_p1a_results.csv')
    ref = ref[(ref.cohort == 'adni') & (ref.scenario == 'native') & (ref.cluster == 'participant') & (ref.kind == 'contrast')]
    for con in ['A minus C', 'P minus C']:
        for b in BUDGETS:
            a = rows[(rows.kind == 'contrast') & (rows.contrast == con) & (rows.budget == b)].iloc[0]
            r = ref[(ref.contrast == con) & (ref.budget == b)].iloc[0]
            d = max(abs(a.estimate_pp - r.estimate_pp), abs(a.ci_low_pp - r.ci_low_pp), abs(a.ci_high_pp - r.ci_high_pp))
            print(f'[2] adni {con} {b:.2f}: library {a.estimate_pp:.2f} ({a.ci_low_pp:.2f}, {a.ci_high_pp:.2f}); '
                  f'original {r.estimate_pp:.2f} ({r.ci_low_pp:.2f}, {r.ci_high_pp:.2f}); max diff {d:.1e}')
            ok &= d < 5e-4      # saved CSV keeps six significant digits

    # 3. committee and locally refit yields against Table 5 / D22
    p2 = pd.read_csv(priv / 'pares_ad_p2_ensemble_baseline_2026-10-14/p2_results.csv')
    corr = pd.read_csv(priv / 'pares_ad_p2_adni_weight_correction_2026-10-15/adni_local_refit_weight_correction.csv')
    corr = corr[corr.weighting == 'config_equal_protocol']
    for name, df in cohorts.items():
        s = src[name]
        w, y = source_weights(s), source_target(s)
        sc = {'CE': predict_head(fit_head(committee_design(s, '_RID'), y, w), committee_design(df))}
        sc.update(crossfit_scores(df, weight_col='review_weight'))
        c = Cohort(name, df, ['configuration_id'] if name == 'adni' else ['configuration_id', 'source_fold'])
        c.set_scores(sc)
        pt, _, _ = run_boot(c, list(sc), 'participant', 1, 1, None, True)
        for mi, m in enumerate(sc):
            scen = 'committee_baseline' if m == 'CE' else 'local_refit'
            table = corr if (name == 'adni' and scen == 'local_refit') else p2
            for bi, b in enumerate(BUDGETS):
                r = table[(table.kind == 'metric') & (table.cohort == name) & (table.scenario == scen) & (table.method == m)
                          & (table.cluster == 'participant') & np.isclose(table.budget, b)]
                lib = 100 * pt[bi, mi, 1] / pt[bi, mi, 0]
                d = abs(lib - float(r.yield_pct.iloc[0]))
                ok &= d < 1e-3
                print(f'[3] {name} {m} {b:.2f}: library yield {lib:.2f}; reported {float(r.yield_pct.iloc[0]):.2f}; diff {d:.1e}')
    print('PARITY OK' if ok else 'PARITY FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
