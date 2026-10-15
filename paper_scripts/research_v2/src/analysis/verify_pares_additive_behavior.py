"""Independent arithmetic, split and closed-form ridge checks of new candidate."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research_v2/outputs/private/pares_ad_additive_behavior_2026-10-09'


def main():
    summary = json.loads((OUT / 'development_summary.json').read_text())
    for name, digest in summary['output_sha256'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    for name, digest in summary['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUT / 'source_fitted_models.joblib').read_bytes()).hexdigest() == summary['source_fit_hash']
    features = pd.read_csv(OUT / 'source_features_private.csv', dtype={'_RID': str})
    predictions = pd.read_csv(OUT / 'development_predictions_private.csv', dtype={'_RID': str})
    records = json.loads((OUT / 'fit_records.json').read_text())
    models = joblib.load(OUT / 'source_fitted_models.joblib')
    keys = ['_RID', 'outer_domain', 'family', 'configuration_id']
    joined = predictions.merge(features, on=keys, validate='one_to_one', suffixes=('', '_source'))
    assert len(joined) == 5184 and joined._RID.nunique() == 650
    assert np.max(np.abs(joined.model_logit - joined.model_logit_source)) < 1e-12
    manual_loss = {}
    methods = ['AD4_LINEAR', 'REF1_LINEAR', 'AD4_ADDITIVE', 'REF1_FLEXIBLE']
    for method in methods:
        loss = (joined['prediction_' + method] - joined.model_logit).abs()
        assert np.allclose(loss, joined['absolute_error_' + method], atol=1e-12)
        manual_loss[method] = pd.Series(loss.to_numpy(), index=joined._RID).groupby(level=0, sort=True).mean().to_numpy()
        assert abs(manual_loss[method].mean() - summary['mae_logit'][method]) < 1e-12
    rng = np.random.default_rng(20261021)
    boot = []
    for _ in range(2000):
        draw = rng.choice(650, 650, replace=True)
        boot.append([float(np.mean(manual_loss[c][draw] - manual_loss['AD4_ADDITIVE'][draw]))
                     for c in ['AD4_LINEAR', 'REF1_FLEXIBLE']])
    difference = float(np.max(np.abs(np.asarray(boot) - np.load(OUT / 'participant_contrasts_bootstrap.npy'))))
    assert difference < 1e-12
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    actual_passes = []
    for i, c in enumerate(['AD4_LINEAR', 'REF1_FLEXIBLE']):
        row = summary['primary_contrasts'][c]
        assert np.allclose(intervals[:, i], row['ci95'], atol=1e-12)
        deltas = joined.groupby('configuration_id').apply(
            lambda frame: (frame['prediction_' + c] - frame.model_logit).abs().mean()
            - (frame.prediction_AD4_ADDITIVE - frame.model_logit).abs().mean(), include_groups=False)
        assert int((deltas > 0).sum()) == row['configurations_improved']
        passed = row['relative_reduction'] >= .1 and intervals[0, i] > 0 and (deltas > 0).sum() >= 7
        assert bool(passed) == row['gate_pass']
        actual_passes.append(passed)
    assert (summary['status'] == 'SOURCE_GATE_FAILED_STOP_CANDIDATE') == (not all(actual_passes))
    max_closed_form_difference = 0.
    for record in records:
        family, config, domain, method = (record[k] for k in ['family', 'configuration_id', 'outer_domain', 'method'])
        train = features[features.family.eq(family) & features.configuration_id.eq(config) & features.outer_domain.ne(domain)]
        val = joined[joined.family.eq(family) & joined.configuration_id.eq(config) & joined.outer_domain.eq(domain)]
        assert set(train._RID).isdisjoint(val._RID)
        assert sorted(train.outer_domain.unique()) == record['training_domains']
        assert len(train) == record['training_people'] and len(val) == record['evaluation_people']
        inner = record['inner_validation']
        if inner:
            assert domain not in {v['inner_domain'] for v in inner}
            alphas = sorted({v['alpha'] for v in inner})
            loss_by_alpha = {a: sum(v['sum_absolute_error'] for v in inner if v['alpha'] == a) /
                            sum(v['people'] for v in inner if v['alpha'] == a) for a in alphas}
            assert record['alpha'] == min(alphas, key=lambda a: (loss_by_alpha[a], -a))
        else:
            assert record['alpha'] == 1.
        cols = [f'C00{i}_severity' for i in range(1, 5)] if method.startswith('AD4') else ['reference_logit']
        model = models[(family, config, domain, method)]
        x = train[cols].to_numpy(float)
        v = val[cols].to_numpy(float)
        if not method.endswith('LINEAR'):
            n_knots = 3 if method == 'AD4_ADDITIVE' else 15
            knots = np.quantile(x, np.linspace(0, 1, n_knots), axis=0)
            for j, basis in enumerate(model[0].bsplines_):
                assert np.allclose(basis.t[3:-3], knots[:, j], atol=1e-10)
            x, v = model[0].transform(x), model[0].transform(v)
            assert x.shape[1] == 16
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd == 0, 1, sd)
        scaler = model[-2]
        assert np.allclose(mu, scaler.mean_, atol=1e-10)
        assert np.allclose(sd, scaler.scale_, atol=1e-10)
        standardized = (x - mu) / sd
        target = train.model_logit.to_numpy()
        coef = np.linalg.solve(standardized.T @ standardized + record['alpha'] * np.eye(x.shape[1]),
                               standardized.T @ (target - target.mean()))
        manual = target.mean() + ((v - mu) / sd) @ coef
        gap = float(np.max(np.abs(manual - val['prediction_' + method])))
        max_closed_form_difference = max(max_closed_form_difference, gap)
        assert gap < 1e-8, (method, gap)
    contribution_cols = [f'C00{i}_behavior_contribution' for i in range(1, 5)]
    reconstruction_difference = float(np.max(np.abs(joined.additive_baseline_logit + joined[contribution_cols].sum(axis=1) - joined.prediction_AD4_ADDITIVE)))
    assert reconstruction_difference < 1e-10
    result = {'status': 'PASS', 'verified_fits': len(records), 'people': 650, 'rows': 5184,
              'maximum_closed_form_prediction_difference': max_closed_form_difference,
              'maximum_contribution_sum_difference': reconstruction_difference,
              'maximum_bootstrap_difference': difference,
              'outer_surrogate_fit_and_tuning_domain_exclusion': True,
              'train_only_knots_scalers_and_ridge_solution': True,
              'scope': 'Saved-data arithmetic and surrogate-fit exclusion; not a full upstream nested validation, inner-CV refit audit, or scientific utility certification.'}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
