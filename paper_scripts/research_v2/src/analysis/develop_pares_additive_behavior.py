"""Protocol-fixed additive behavior candidate, conditional development CV only."""
from pathlib import Path
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from research_v2.src.analysis.confirm_pares_ad_state_confirmation import (
    load_development_joined, DEV_PREDICTIONS, BASELINE_CONCEPTS, ASSIGNMENTS,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import CORE_CONCEPTS, logit

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'research_v2/protocols/pares_ad_additive_behavior_redesign_2026-10-09.json'
OUT = ROOT / 'research_v2/outputs/private/pares_ad_additive_behavior_2026-10-09'
METHODS = ['AD4_LINEAR', 'REF1_LINEAR', 'AD4_ADDITIVE', 'REF1_FLEXIBLE']
AD_COLUMNS = [f'{c}_severity' for c in CORE_CONCEPTS]
GRID = [0.1, 1., 10., 100.]
KEYS = ['_RID', 'outer_domain', 'family', 'configuration_id']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def columns(method):
    return AD_COLUMNS if method.startswith('AD4') else ['reference_logit']


def estimator(method, alpha):
    if method.endswith('LINEAR'):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return make_pipeline(
        SplineTransformer(n_knots=3 if method == 'AD4_ADDITIVE' else 15,
                          degree=3, knots='quantile', include_bias=False,
                          extrapolation='constant'),
        StandardScaler(), Ridge(alpha=alpha),
    )


def select_alpha(train, method):
    """All basis positions, scalers and coefficients fit anew inside each fold."""
    if method.endswith('LINEAR'):
        return 1., []
    cols = columns(method)
    losses = []
    for held in sorted(train.outer_domain.unique()):
        fit = train[train.outer_domain.ne(held)]
        val = train[train.outer_domain.eq(held)]
        assert set(fit._RID).isdisjoint(val._RID)
        assert fit.outer_domain.nunique() >= 2
        for alpha in GRID:
            model = estimator(method, alpha).fit(fit[cols].to_numpy(float), fit.model_logit.to_numpy(float))
            assert model[0].n_features_out_ == 16
            pred = model.predict(val[cols].to_numpy(float))
            losses.append({'inner_domain': held, 'alpha': alpha, 'people': len(val),
                           'sum_absolute_error': float(np.abs(pred - val.model_logit).sum())})
    means = {alpha: sum(row['sum_absolute_error'] for row in losses if row['alpha'] == alpha)
             / sum(row['people'] for row in losses if row['alpha'] == alpha) for alpha in GRID}
    chosen = min(GRID, key=lambda alpha: (means[alpha], -alpha))
    return chosen, losses


def additive_contributions(model, x, reference):
    """Exact output decomposition relative to the training feature median."""
    basis = model[0].transform(x)
    base = model[0].transform(reference.reshape(1, -1))
    changes = (basis - base) / model[1].scale_ * model[2].coef_
    contributions = changes.reshape(len(x), 4, 4).sum(axis=2)
    intercept = float(model.predict(reference.reshape(1, -1))[0])
    assert np.allclose(intercept + contributions.sum(axis=1), model.predict(x), atol=1e-10, rtol=0)
    return intercept, contributions


def summarize(predictions):
    losses = []
    for method in METHODS:
        col = 'absolute_error_' + method
        predictions[col] = np.abs(predictions['prediction_' + method] - predictions.model_logit)
        losses.append(col)
    people = predictions.groupby('_RID', sort=True)[losses].mean()
    mae = {m: float(people['absolute_error_' + m].mean()) for m in METHODS}
    group_rows = []
    for scope, field in [('configuration', 'configuration_id'), ('domain', 'outer_domain')]:
        for value, frame in predictions.groupby(field, sort=True):
            means = frame.groupby('_RID')[losses].mean().mean()
            group_rows.append({'scope': scope, 'value': value, 'people': frame._RID.nunique(),
                               'prediction_rows': len(frame),
                               **{'mae_' + m: float(means['absolute_error_' + m]) for m in METHODS}})
    groups = pd.DataFrame(group_rows)
    config = groups[groups.scope.eq('configuration')]
    comparator_names = ['AD4_LINEAR', 'REF1_FLEXIBLE']
    contrasts = np.column_stack([people['absolute_error_' + name] - people.absolute_error_AD4_ADDITIVE
                                 for name in comparator_names])
    rng = np.random.default_rng(20261021)
    boot = np.empty((2000, 2))
    for i in range(len(boot)):
        boot[i] = contrasts[rng.integers(0, len(people), len(people))].mean(axis=0)
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    result = {}
    for i, comparator in enumerate(comparator_names):
        gain = mae[comparator] - mae['AD4_ADDITIVE']
        relative = gain / mae[comparator]
        directions = int((config.mae_AD4_ADDITIVE < config['mae_' + comparator]).sum())
        result[comparator] = {'mae_reduction': gain, 'relative_reduction': relative,
                              'ci95': intervals[:, i].tolist(), 'configurations_improved': directions,
                              'gate_pass': bool(relative >= .1 and intervals[0, i] > 0 and directions >= 7)}
    return mae, result, groups, boot


def main():
    started = time.perf_counter()
    if (OUT / 'run_started.json').exists():
        raise RuntimeError('Run already started; inspect its state and preserve existing outputs before retrying.')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['alpha_grid'] == GRID
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in [PROTOCOL, Path(__file__), DEV_PREDICTIONS, BASELINE_CONCEPTS, ASSIGNMENTS]}
    (OUT / 'run_started.json').write_text(json.dumps({'source_sha256': provenance, 'stage': 'source_development_only'}, indent=2))
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    development, _, _, _ = load_development_joined()
    development['reference_logit'] = logit(development.pathology_probability_AD)
    keep = KEYS + AD_COLUMNS + ['reference_logit', 'model_logit']
    data = development[keep].copy().sort_values(KEYS).reset_index(drop=True)
    assert data._RID.nunique() == 650 and len(data) == 5184
    assert not data.duplicated(KEYS).any()
    assert data.groupby('_RID').outer_domain.nunique().max() == 1
    data.to_csv(OUT / 'source_features_private.csv', index=False, float_format='%.17g')
    fits = {}
    parts, records = [], []
    for (family, config), group in data.groupby(['family', 'configuration_id'], sort=True):
        for domain in sorted(group.outer_domain.unique()):
            train = group[group.outer_domain.ne(domain)].copy()
            val = group[group.outer_domain.eq(domain)].copy()
            assert train._RID.nunique() == len(train)
            assert set(train._RID).isdisjoint(val._RID)
            block = val[KEYS + ['model_logit']].copy()
            for method in METHODS:
                alpha, inner = select_alpha(train, method)
                cols = columns(method)
                x_train, x_val = train[cols].to_numpy(float), val[cols].to_numpy(float)
                model = estimator(method, alpha).fit(x_train, train.model_logit.to_numpy(float))
                predicted = model.predict(x_val)
                assert np.isfinite(predicted).all()
                block['prediction_' + method] = predicted
                record = {'family': family, 'configuration_id': config, 'outer_domain': domain,
                          'method': method, 'alpha': alpha, 'training_people': len(train),
                          'evaluation_people': len(val), 'inner_validation': inner,
                          'basis_features': int(model[-1].n_features_in_),
                          'iterations': None, 'training_domains': sorted(train.outer_domain.unique())}
                if method == 'AD4_ADDITIVE':
                    ref = np.median(x_train, axis=0)
                    intercept, contributions = additive_contributions(model, x_val, ref)
                    block['additive_baseline_logit'] = intercept
                    for i, concept in enumerate(CORE_CONCEPTS):
                        block[concept + '_behavior_contribution'] = contributions[:, i]
                    record['contribution_reference_median'] = ref.tolist()
                fits[(family, config, domain, method)] = model
                records.append(record)
            parts.append(block)
        print(f'Completed nested source surrogate evaluation: {family}/{config}', flush=True)
    predictions = pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    assert len(predictions) == len(data)
    mae, contrasts, groups, boot = summarize(predictions)
    gate = all(v['gate_pass'] for v in contrasts.values())
    predictions.to_csv(OUT / 'development_predictions_private.csv', index=False, float_format='%.17g')
    groups.to_csv(OUT / 'group_results_aggregate.csv', index=False)
    np.save(OUT / 'participant_contrasts_bootstrap.npy', boot)
    joblib.dump(fits, OUT / 'source_fitted_models.joblib')
    (OUT / 'fit_records.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    result = {
        'status': 'SOURCE_GATE_PASSED_EXPLORATORY_PAIR_STAGE_ELIGIBLE' if gate else 'SOURCE_GATE_FAILED_STOP_CANDIDATE',
        'people': 650, 'prediction_rows': len(predictions), 'configuration_domain_groups': len(parts),
        'mae_logit': mae, 'primary_contrasts': contrasts,
        'source_fit_hash': sha(OUT / 'source_fitted_models.joblib'), 'source_sha256': provenance,
        'output_sha256': {name: sha(OUT / name) for name in ['source_features_private.csv', 'development_predictions_private.csv', 'fit_records.json', 'group_results_aggregate.csv', 'participant_contrasts_bootstrap.npy']},
        'confirmation_or_pair_evaluated': False,
        'interpretation': 'Conditional method-development result for fixed upstream artifacts; no claim of independent end-to-end validation, causal explanation or reader utility.',
        'runtime_seconds': time.perf_counter() - started,
    }
    (OUT / 'development_summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'people', 'prediction_rows', 'mae_logit', 'primary_contrasts', 'runtime_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
