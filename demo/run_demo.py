"""End-to-end demonstration on synthetic data: fit review heads on 'source' predictions, score a shifted 'target'
cohort, evaluate fixed-budget review yield with participant bootstrap, and add the committee and local-refit
comparators. Run from the repository root: python demo/run_demo.py [B]."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pares_ad import (fit_head, predict_head, design, extras, committee_design, source_weights, source_target,
                      crossfit_scores, Cohort, run_boot, summarize)
from make_synthetic import make_reference, make_cohort


def main(B=200):
    ref = make_reference()
    src = make_cohort(600, 0.40, 0.0, 11, ref)
    tgt = make_cohort(900, 0.25, 0.8, 12, ref)
    w, y = source_weights(src), source_target(src)
    heads = {'C': fit_head(design(src), y, w), 'A': fit_head(design(src, extras(src, 'A')), y, w),
             'P': fit_head(design(src, extras(src, 'P')), y, w),
             'CE': fit_head(committee_design(src), y, w)}
    scores = {'C': predict_head(heads['C'], design(tgt)), 'A': predict_head(heads['A'], design(tgt, extras(tgt, 'A'))),
              'P': predict_head(heads['P'], design(tgt, extras(tgt, 'P'))), 'CE': predict_head(heads['CE'], committee_design(tgt))}
    scores.update(crossfit_scores(tgt))
    coh = Cohort('synthetic_target', tgt, ['configuration_id'])
    coh.set_scores(scores)
    methods = ['C', 'A', 'P', 'CE', 'Cx', 'Px']
    pairs = [('A', 'C'), ('P', 'C'), ('CE', 'C'), ('P', 'CE'), ('Px', 'Cx')]
    pt, ptot, boots = run_boot(coh, methods, 'participant', B, 20261014)
    rows = pd.DataFrame(summarize('synthetic_target', 'demo', methods, pt, ptot, boots, pairs, 'participant', B))
    pd.set_option('display.width', 200)
    print('Yield (%) of classifier errors among reviewed predictions')
    print(rows[rows.kind == 'metric'].pivot(index='budget', columns='method', values='yield_pct').round(1))
    print('\nDifferences, percentage points (95% participant-cluster interval)')
    c = rows[rows.kind == 'contrast'].copy()
    c['text'] = c.apply(lambda r: f'{r.estimate_pp:.1f} ({r.ci_low_pp:.1f}, {r.ci_high_pp:.1f})', axis=1)
    print(c.pivot(index='contrast', columns='budget', values='text').to_string())
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
