"""Unit tests. Run from the repository root: python -m pytest tests  (or: python tests/test_pares_ad.py)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'demo'))
from pares_ad import (fit_signed_reference, reference_logit, conformal_absolute_quantile, fit_output_decomposition,
                      fit_head, predict_head, design, extras, opposition, source_weights, source_target,
                      committee_features, Cohort, run_boot, fixed_budget_yield, case_report, reliability_state)
from make_synthetic import make_reference, make_cohort, _features


def test_reference_is_sign_constrained_and_additive():
    rng = np.random.default_rng(0)
    x, y = _features(800, 0.4, rng)
    b0, coef = fit_signed_reference(x, y)
    assert (coef >= 0).all() and coef.sum() > 0
    logit, contrib = reference_logit(x, b0, coef)
    assert np.allclose(logit, b0 + contrib.sum(1), atol=1e-12)
    # opposite-sign data must be clipped at zero rather than flipped
    _, coef2 = fit_signed_reference(-x, y)
    assert (coef2 >= 0).all() and coef2.sum() < coef.sum()


def test_head_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    src = make_cohort(300, 0.4, 0.0, 5, make_reference())
    w, y = source_weights(src), source_target(src)
    X = design(src, extras(src, 'P'))
    ours = predict_head(fit_head(X, y, w), X)
    sk = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, tol=1e-10))
    sk.fit(X, y, logisticregression__sample_weight=w)
    assert np.max(np.abs(ours - sk.predict_proba(X)[:, 1])) < 1e-4


def test_decomposition_matches_sklearn_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    S = rng.normal(size=(200, 4))
    A = np.column_stack([(rng.integers(0, 3, 200) == k).astype(float) for k in range(3)])
    y = S @ [1.0, 0.5, 0.0, 0.3] + A @ [0.2, -0.1, 0.0] + rng.normal(size=200) * 0.2
    dec = fit_output_decomposition(S, A, y)
    m1 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(S, y)
    d_ref = m1.predict(S)
    m2 = Ridge(alpha=1.0).fit(A, y - d_ref)
    disease, acq = dec.components(S, A)
    assert np.allclose(disease, d_ref, atol=1e-9) and np.allclose(acq, m2.predict(A), atol=1e-9)


def test_conformal_quantile_and_state():
    assert conformal_absolute_quantile(np.arange(1, 100), 0.90) == 90.0   # ceil(100*0.9) = 90th smallest
    assert reliability_state(False, 0.0, 1.0, True)[0] == 'UNEXPLAINED_ABSTAIN'
    assert reliability_state(True, 0.5, 0.1, False)[0] == 'ACQUISITION_SENSITIVE'
    assert reliability_state(True, 0.1, 0.5, True)[0] == 'PATHOLOGY_CONFLICT'


def test_opposition_and_disagreement():
    p = np.array([0.9, 0.2, 0.7, 0.4])
    q = np.array([0.1, 0.8, 0.6, 0.3])
    assert np.allclose(opposition(p, q), [0.9, 0.8, 0.4, 0.3])      # classifier AD -> 1-q; classifier CN -> q
    df = pd.DataFrame(dict(model_probability_AD=p, pathology_probability_AD=q))
    assert extras(df, 'A').tolist() == [True, True, False, False]


def test_fixed_budget_engine_matches_brute_force():
    tgt = make_cohort(120, 0.3, 0.3, 7, make_reference())
    coh = Cohort('t', tgt, ['configuration_id'])
    s = np.random.default_rng(0).random(len(tgt))
    coh.set_scores({'S': s})
    pt, ptot, boots = run_boot(coh, ['S'], 'participant', 40, 1)
    for bi, b in enumerate([.05, .10, .20]):
        ys = []
        for _, g in tgt.assign(s=s).groupby('configuration_id'):
            e = (g.model_probability_AD >= .5).astype(int) != g.AD_label
            ys.append(fixed_budget_yield(g.s.to_numpy(), e.to_numpy(float), b))
        # equal-size groups with equal weights: pooled yield is the mean of group yields
        assert abs(pt[bi, 0, 1] / pt[bi, 0, 0] - np.mean(ys)) < 1e-9
    assert boots.shape == (40, 3, 1) and np.isfinite(boots).all()


def test_committee_features_are_participant_level():
    tgt = make_cohort(50, 0.3, 0.0, 3, make_reference())
    sd, vote = committee_features(tgt)
    g = pd.DataFrame(dict(k=tgt.participant_key, sd=sd, vote=vote)).groupby('k').nunique()
    assert (g.sd == 1).all() and (g.vote == 1).all() and (vote >= 0).all() and (vote <= 1).all()


def test_case_report_withholds_contributions():
    kw = dict(case_id='X', classifier_probability=0.77, reference_intercept=-2.0, severities=[1, 1, 1, 1], coefficients=[.3, .3, .3, .3])
    assert 'WITHHELD' in case_report(residual=2.0, residual_bound=1.0, **kw)
    out = case_report(residual=0.1, residual_bound=1.0, **kw)
    assert 'WITHHELD' not in out and 'Intercept' in out


def test_demo_runs():
    import run_demo
    rows = run_demo.main(40)
    assert len(rows) > 0 and set(rows.kind) == {'metric', 'contrast'}


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in fns:
        f()
        print('ok', f.__name__)
