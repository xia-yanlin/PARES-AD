"""Fixed linear change-target learning on isolated source natural pairs."""
from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from research_v2.src.analysis.confirm_pares_ad_independent import (
    RAW_FS, BASELINE_CONCEPTS, ASSIGNMENTS, MANIFEST, MAPPING, REGISTRY, run_black_boxes,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS, CONCEPT_ORIENTATION, fit_monotone_pathology_model, logit,
)
from research_v2.src.audit.concepts import build_concept_target
from research_v2.src.execution.fw5_common import MEMBERSHIP_PATH

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research_v2/outputs/private/pares_ad_source_pair_development_2026-10-09'
PRIOR = ROOT / 'research_v2/outputs/private/pares_ad_teacher_consistency_2026-10-09'
PROTOCOL = ROOT / 'research_v2/protocols/pares_ad_direct_change_learning_2026-10-09.json'
AD_COLS = [f'{c}_severity' for c in CORE_CONCEPTS]
DELTA_COLS = ['delta_' + c for c in AD_COLS]
METHODS = ['LEVEL_AD4', 'DELTA_AD4', 'DELTA_REF1']
COMPARATORS = ['LEVEL_AD4', 'DELTA_REF1']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(frame):
    for m in METHODS:
        frame['loss_' + m] = (frame['prediction_' + m] - frame.observed_delta).abs()
    loss_cols = ['loss_' + m for m in METHODS]
    people = frame.groupby('_RID', sort=True)[loss_cols].mean()
    mae = {m: float(people['loss_' + m].mean()) for m in METHODS}
    group_rows = []
    for scope, col in [('configuration', 'configuration_id'), ('domain', 'actual_domain')]:
        for name, g in frame.groupby(col):
            avg = g.groupby('_RID')[loss_cols].mean().mean()
            group_rows.append({'scope': scope, 'value': name, 'people': g._RID.nunique(),
                               **{'mae_' + m: float(avg['loss_' + m]) for m in METHODS}})
    groups = pd.DataFrame(group_rows)
    configs = groups[groups.scope.eq('configuration')]
    rng = np.random.default_rng(20261023)
    values = np.column_stack([people['loss_' + m] - people.loss_DELTA_AD4 for m in COMPARATORS])
    boot = np.empty((2000, 2))
    for b in range(2000):
        boot[b] = values[rng.integers(0, len(people), len(people))].mean(axis=0)
    ci = np.quantile(boot, [.025, .975], axis=0)
    contrasts = {}
    for i, m in enumerate(COMPARATORS):
        gain = mae[m] - mae['DELTA_AD4']
        relative = gain / mae[m]
        direction = int((configs.mae_DELTA_AD4 < configs['mae_' + m]).sum())
        contrasts[m] = {'absolute_mae_reduction': gain, 'relative_reduction': relative,
                        'ci95': ci[:, i].tolist(), 'configurations_improved': direction,
                        'gate_pass': bool(relative >= .1 and ci[0, i] > 0 and direction >= 7)}
    return mae, contrasts, groups, boot


def followup_features(baseline, followup, common, domain):
    train = baseline[baseline.acquisition_domain.ne(domain)].reset_index(drop=True)
    evaluation = pd.concat([baseline, followup], ignore_index=True)
    x_train, x_eval = [], []
    for c in CORE_CONCEPTS:
        target = build_concept_target(c, train, evaluation)
        assert target.train_valid.all() and target.evaluation_valid.all()
        x_train.append(CONCEPT_ORIENTATION[c] * target.train_values)
        x_eval.append(CONCEPT_ORIENTATION[c] * target.evaluation_values)
    x_train, x_eval = np.column_stack(x_train), np.column_stack(x_eval)
    intercept, beta = fit_monotone_pathology_model(x_train, train.AD_label.to_numpy(int), .1)
    q = logit(expit(intercept + x_eval @ beta))
    old = baseline[['_RID']].merge(common[common.teacher_domain.eq(domain)], on='_RID', validate='one_to_one')
    gap = max(float(np.max(np.abs(x_eval[:len(baseline)] - old[AD_COLS].to_numpy()))),
              float(np.max(np.abs(q[:len(baseline)] - old.reference_logit))))
    assert gap < 1e-9
    result = followup[['_RID', 'actual_domain']].copy()
    result['teacher_domain'] = domain
    result[AD_COLS] = x_eval[len(baseline):]
    result['reference_logit'] = q[len(baseline):]
    return result, gap


def main():
    if (OUT / 'development_started.json').exists():
        raise RuntimeError('Prior development run exists; inspect before retrying.')
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    torch.set_num_threads(1)
    feasibility = json.loads((OUT / 'feasibility.json').read_text())
    assert feasibility['status'] == 'FEASIBLE_FOR_FROZEN_METHOD_PROTOCOL'
    assert sha(OUT / 'pair_manifest_private.csv') == feasibility['pair_manifest_sha256']
    prior_summary = json.loads((PRIOR / 'development_summary.json').read_text())
    for n in ['teacher_predictions_private.csv', 'common_features_private.csv', 'source_fitted_models.joblib']:
        assert sha(PRIOR / n) == prior_summary['output_sha256'][n]
    files = [PROTOCOL, Path(__file__), RAW_FS, BASELINE_CONCEPTS, ASSIGNMENTS, MANIFEST,
             MAPPING, REGISTRY, MEMBERSHIP_PATH, OUT / 'pair_manifest_private.csv',
             PRIOR / 'teacher_predictions_private.csv', PRIOR / 'common_features_private.csv', PRIOR / 'source_fitted_models.joblib']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in files}
    (OUT / 'development_started.json').write_text(json.dumps({'source_sha256': sources, 'stage': 'source_only'}, indent=2))
    pairs = pd.read_csv(OUT / 'pair_manifest_private.csv', dtype={'_RID': str})
    baseline = pd.read_csv(BASELINE_CONCEPTS, low_memory=False, dtype={'RID': str})
    assignment = pd.read_csv(ASSIGNMENTS, dtype={'RID': str})
    baseline = baseline.merge(assignment[['RID', 'acquisition_domain']], on='RID', validate='one_to_one')
    baseline['_RID'] = baseline.RID
    raw = pd.read_csv(RAW_FS, low_memory=False)
    followup = raw.iloc[pairs.followup_raw_row.to_numpy(int)].copy().reset_index(drop=True)
    assert np.array_equal(pd.to_numeric(followup.RID).astype('Int64').astype(str), pairs._RID)
    followup['_RID'] = pairs._RID.to_numpy()
    followup['actual_domain'] = pairs.outer_domain.to_numpy()
    # The inference helper requires a label column but does not use it. These
    # followups have no diagnosis outcome in this study, and it is dropped.
    followup['AD_label'] = -1
    common = pd.read_csv(PRIOR / 'common_features_private.csv', dtype={'_RID': str})
    anchors = pd.read_csv(PRIOR / 'teacher_predictions_private.csv', dtype={'_RID': str})
    manifest = pd.read_csv(MANIFEST)
    mapping = json.loads(MAPPING.read_text())['nodes']
    registry = json.loads(REGISTRY.read_text())
    replay, features, replay_checks = [], [], []
    for domain, jobs in manifest.groupby('outer_domain', sort=True):
        routed = followup.copy()
        routed['outer_domain'] = domain
        predicted = run_black_boxes(routed, jobs, mapping, registry).drop(columns=['AD_label'])
        predicted['followup_logit'] = logit(predicted.model_probability_AD)
        replay.append(predicted)
        ft, gap = followup_features(baseline, followup, common, domain)
        features.append(ft)
        replay_checks.append({'teacher_domain': domain, 'max_frozen_feature_replay_difference': gap})
        print(f'Completed frozen-teacher source followup replay: {domain}', flush=True)
    replay = pd.concat(replay, ignore_index=True)
    ft = pd.concat(features, ignore_index=True)
    assert len(replay) == len(pairs) * 58 and replay.seed_count.eq(5).all()
    replay.to_csv(OUT / 'followup_teacher_predictions_private.csv', index=False, float_format='%.17g')
    ft.to_csv(OUT / 'followup_features_private.csv', index=False, float_format='%.17g')
    anchor_features = common[common._RID.isin(pairs._RID)]
    delta = anchor_features.merge(ft, on=['_RID', 'teacher_domain', 'actual_domain'], suffixes=('_a', '_b'), validate='one_to_one')
    for c in AD_COLS + ['reference_logit']:
        delta['delta_' + c] = delta[c + '_b'] - delta[c + '_a']
    delta = delta[['_RID', 'teacher_domain', 'actual_domain'] + DELTA_COLS + ['delta_reference_logit']]
    key = ['_RID', 'outer_domain', 'family', 'configuration_id']
    joined = replay.merge(anchors[key + ['teacher_logit']], on=key, validate='one_to_one')
    joined['observed_delta'] = joined.followup_logit - joined.teacher_logit
    joined = joined.merge(delta, left_on=['_RID', 'outer_domain'], right_on=['_RID', 'teacher_domain'], validate='many_to_one')
    joined.to_csv(OUT / 'all_teacher_pair_deltas_private.csv', index=False, float_format='%.17g')
    membership = pd.read_csv(MEMBERSHIP_PATH, dtype={'RID': str})
    old_fits = joblib.load(PRIOR / 'source_fitted_models.joblib')
    fits, records, evaluation_rows = {}, [], []
    for (family, config, domain), group in joined.groupby(['family', 'configuration_id', 'outer_domain'], sort=True):
        train, val = group[group.actual_domain.ne(domain)], group[group.actual_domain.eq(domain)].copy()
        assert set(train._RID).isdisjoint(val._RID)
        train_ids = set(membership.loc[membership.outer_domain.eq(domain) & membership.outer_role.eq('train_pool'), 'RID'])
        test_ids = set(membership.loc[membership.outer_domain.eq(domain) & membership.outer_role.eq('test'), 'RID'])
        assert set(train._RID) <= train_ids and set(val._RID) <= test_ids
        assert len(train) >= 50
        old_model = old_fits[(family, config, domain, 'MATCHED_AD4')]
        val['prediction_LEVEL_AD4'] = val[DELTA_COLS].to_numpy() / old_model[0].scale_ @ old_model[1].coef_
        for name, cols in [('DELTA_AD4', DELTA_COLS), ('DELTA_REF1', ['delta_reference_logit'])]:
            model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1., fit_intercept=False))
            model.fit(train[cols].to_numpy(float), train.observed_delta.to_numpy(float))
            val['prediction_' + name] = model.predict(val[cols].to_numpy(float))
            assert np.all(model.predict(np.zeros((1, len(cols)))) == 0)
            fits[(family, config, domain, name)] = model
            records.append({'family': family, 'configuration_id': config, 'outer_domain': domain,
                            'method': name, 'features': cols, 'training_people': len(train), 'evaluation_people': len(val),
                            'alpha': 1., 'fit_intercept': False, 'with_mean': False})
        evaluation_rows.append(val)
    evaluated = pd.concat(evaluation_rows, ignore_index=True)
    assert evaluated._RID.nunique() == 218
    mae, contrasts, aggregate, boot = summarize(evaluated)
    evaluated.to_csv(OUT / 'development_predictions_private.csv', index=False, float_format='%.17g')
    aggregate.to_csv(OUT / 'group_results_aggregate.csv', index=False)
    np.save(OUT / 'participant_contrasts_bootstrap.npy', boot)
    joblib.dump(fits, OUT / 'source_delta_models.joblib')
    (OUT / 'fit_records.json').write_text(json.dumps(records, indent=2) + '\n')
    (OUT / 'feature_replay_checks.json').write_text(json.dumps(replay_checks, indent=2) + '\n')
    output_names = ['followup_teacher_predictions_private.csv', 'followup_features_private.csv',
                    'all_teacher_pair_deltas_private.csv', 'development_predictions_private.csv', 'group_results_aggregate.csv',
                    'participant_contrasts_bootstrap.npy', 'source_delta_models.joblib', 'fit_records.json', 'feature_replay_checks.json']
    result = {'status': 'SOURCE_GATE_PASSED_PAIR_STAGE_ELIGIBLE' if all(c['gate_pass'] for c in contrasts.values()) else 'SOURCE_GATE_FAILED_STOP_CANDIDATE',
              'people': 218, 'evaluated_configuration_pairs': len(evaluated), 'mae_logit_change': mae,
              'primary_contrasts': contrasts, 'confirmation_evaluated': False,
              'source_sha256': sources, 'output_sha256': {n: sha(OUT / n) for n in output_names},
              'scope': 'Exploratory method development with held-domain participants; fixed-fit uncertainty, no clinical or causal claim.'}
    (OUT / 'development_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'people', 'evaluated_configuration_pairs', 'mae_logit_change', 'primary_contrasts']}, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
