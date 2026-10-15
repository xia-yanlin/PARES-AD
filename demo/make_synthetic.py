"""Synthetic cohorts with the schema used by pares_ad (no real data).

Four oriented structural severities separate AD from CN. Nine 'classifiers' see the same signal plus individual
noise; the target cohort has a different AD share and a shifted classifier bias, as in a new site.
"""
import numpy as np
import pandas as pd
from pares_ad import fit_signed_reference, reference_logit

MU_AD = np.array([1.2, 1.0, 0.9, 1.1])
CONFIGS = [f'cfg{i}' for i in range(9)]


def _features(n, ad_share, rng):
    y = (rng.random(n) < ad_share).astype(int)
    x = rng.normal(size=(n, 4)) + y[:, None] * MU_AD
    return x, y


def make_reference(seed=1):
    rng = np.random.default_rng(seed)
    x, y = _features(1500, 0.4, rng)
    return fit_signed_reference(x, y)


def make_cohort(n, ad_share, bias, seed, reference, n_centers=1):
    rng = np.random.default_rng(seed)
    x, y = _features(n, ad_share, rng)
    intercept, coef = reference
    ref_logit, _ = reference_logit(x, intercept, coef)
    p_ref = 1 / (1 + np.exp(-ref_logit))
    signal = rng.normal(size=n) * 0.7 + (2 * y - 1) * 2.0     # participant-level classifier view of the diagnosis
    keys = [f'p{seed}_{i:05d}' for i in range(n)]
    centers = rng.integers(0, n_centers, n)
    rows = []
    for ci, cfg in enumerate(CONFIGS):
        logit = signal + rng.normal(size=n) * 1.2 + bias + 0.15 * (ci - 4)
        rows.append(pd.DataFrame(dict(participant_key=keys, row_key=keys, configuration_id=cfg, center=centers.astype(str),
                                      AD_label=y, model_probability_AD=1 / (1 + np.exp(-logit)),
                                      pathology_probability_AD=p_ref, review_weight=1 / len(CONFIGS))))
    return pd.concat(rows, ignore_index=True)
