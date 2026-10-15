"""Test exact-teacher supervision with fixed linear surrogate complexity."""
from pathlib import Path
import hashlib
import json
import time
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
    BASELINE_CONCEPTS, ASSIGNMENTS, MANIFEST, MAPPING, REGISTRY, JOBS_ROOT, run_black_boxes,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS, CONCEPT_ORIENTATION, fit_monotone_pathology_model, logit,
)
from research_v2.src.audit.concepts import build_concept_target
from research_v2.src.execution.fw5_common import MEMBERSHIP_PATH

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'research_v2/protocols/pares_ad_teacher_consistency_2026-10-09.json'
OUT = ROOT / 'research_v2/outputs/private/pares_ad_teacher_consistency_2026-10-09'
OLD = ROOT / 'research_v2/outputs/private/pares_ad_additive_behavior_2026-10-09'
SOURCE = ROOT / 'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08/source_full_private.csv'
KEYS = ['_RID', 'outer_domain', 'family', 'configuration_id']
AD_COLS = [f'{c}_severity' for c in CORE_CONCEPTS]
METHODS = ['HISTORICAL_AD4', 'MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']
COMPARATORS = ['MIXED_AD4', 'MATCHED_REF1', 'HISTORICAL_AD4']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_common_features(baseline, domain):
    train = baseline[baseline.acquisition_domain.ne(domain)].reset_index(drop=True)
    evaluation = baseline.reset_index(drop=True)
    train_features, all_features = [], []
    for concept in CORE_CONCEPTS:
        target = build_concept_target(concept, train, evaluation)
        assert target.train_valid.all() and target.evaluation_valid.all()
        train_features.append(CONCEPT_ORIENTATION[concept] * target.train_values)
        all_features.append(CONCEPT_ORIENTATION[concept] * target.evaluation_values)
    x_train, x_all = np.column_stack(train_features), np.column_stack(all_features)
    intercept, beta = fit_monotone_pathology_model(x_train, train.AD_label.to_numpy(int), .1)
    frame = evaluation[['_RID', 'acquisition_domain']].rename(columns={'acquisition_domain': 'actual_domain'}).copy()
    frame['teacher_domain'] = domain
    frame[AD_COLS] = x_all
    frame['reference_logit'] = logit(expit(intercept + x_all @ beta))
    return frame


def summarize(rows):
    for name in METHODS:
        rows['loss_' + name] = np.abs(rows['prediction_' + name] - rows.model_logit)
    loss_cols = ['loss_' + m for m in METHODS]
    people = rows.groupby('_RID', sort=True)[loss_cols].mean()
    mae = {m: float(people['loss_' + m].mean()) for m in METHODS}
    aggregates = []
    for scope, field in [('configuration', 'configuration_id'), ('domain', 'outer_domain')]:
        for value, group in rows.groupby(field, sort=True):
            means = group.groupby('_RID')[loss_cols].mean().mean()
            aggregates.append({'scope': scope, 'value': value, 'people': group._RID.nunique(),
                               **{'mae_' + m: float(means['loss_' + m]) for m in METHODS}})
    aggregate = pd.DataFrame(aggregates)
    configs = aggregate[aggregate.scope.eq('configuration')]
    contrast_values = np.column_stack([people['loss_' + m] - people.loss_MATCHED_AD4 for m in COMPARATORS])
    rng = np.random.default_rng(20261022)
    boot = np.empty((2000, len(COMPARATORS)))
    for b in range(len(boot)):
        boot[b] = contrast_values[rng.integers(0, len(people), len(people))].mean(axis=0)
    intervals = np.quantile(boot, [.025, .975], axis=0)
    contrasts = {}
    for i, name in enumerate(COMPARATORS):
        gain = mae[name] - mae['MATCHED_AD4']
        relative = gain / mae[name]
        directions = int((configs.mae_MATCHED_AD4 < configs['mae_' + name]).sum())
        contrasts[name] = {'absolute_mae_reduction': gain, 'relative_reduction': relative,
                           'ci95': intervals[:, i].tolist(), 'configurations_improved': directions,
                           'gate_pass': bool(relative >= .1 and intervals[0, i] > 0 and directions >= 7)}
    return mae, contrasts, aggregate, boot


def main():
    started = time.perf_counter()
    if (OUT / 'run_started.json').exists():
        raise RuntimeError('A prior run exists; inspect it rather than automatically restarting.')
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [PROTOCOL, Path(__file__), BASELINE_CONCEPTS, ASSIGNMENTS, MANIFEST, MAPPING,
             REGISTRY, MEMBERSHIP_PATH, SOURCE, OLD / 'development_predictions_private.csv']
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    (OUT / 'run_started.json').write_text(json.dumps({'source_sha256': provenance, 'stage': 'development_only'}, indent=2))
    source = pd.read_csv(SOURCE, dtype={'_RID': str})
    source['model_logit'] = logit(source.model_probability_AD)
    assert len(source) == 5184 and source._RID.nunique() == 650
    baseline = pd.read_csv(BASELINE_CONCEPTS, dtype={'RID': str}, low_memory=False)
    assignments = pd.read_csv(ASSIGNMENTS, dtype={'RID': str})
    baseline = baseline.merge(assignments[['RID', 'acquisition_domain']], on='RID', validate='one_to_one')
    baseline['_RID'] = baseline.RID
    manifest = pd.read_csv(MANIFEST, dtype={'seed': int})
    membership = pd.read_csv(MEMBERSHIP_PATH, dtype={'RID': str})
    mapping = json.loads(MAPPING.read_text())['nodes']
    registry = json.loads(REGISTRY.read_text())
    replay_parts, common_parts, split_receipts = [], [], []
    for domain, jobs in manifest.groupby('outer_domain', sort=True):
        members = membership[membership.outer_domain.eq(domain)]
        train_ids = set(members.loc[members.outer_role.eq('train_pool'), 'RID'])
        test_ids = set(members.loc[members.outer_role.eq('test'), 'RID'])
        assert train_ids and test_ids and train_ids.isdisjoint(test_ids)
        observed_ids = set(source.loc[source.outer_domain.eq(domain), '_RID'])
        assert observed_ids <= test_ids
        assert set(baseline.loc[baseline.acquisition_domain.ne(domain), '_RID']).isdisjoint(test_ids)
        routing = baseline.copy()
        # Inference helper selects rows by checkpoint domain; true acquisition
        # domain stays in acquisition_domain and in saved common-feature rows.
        routing['outer_domain'] = domain
        replay = run_black_boxes(routing, jobs, mapping, registry).drop(columns=['AD_label'])
        replay['teacher_logit'] = logit(replay.model_probability_AD)
        replay_parts.append(replay)
        common_parts.append(source_common_features(baseline, domain))
        split_receipts.append({'teacher_domain': domain, 'teacher_train_pool_people': len(train_ids),
                               'teacher_test_people': len(test_ids), 'development_test_people': len(observed_ids),
                               'teacher_training_test_overlap': 0})
        print(f'Completed exact-teacher source replay and common transforms: {domain}', flush=True)
    replay = pd.concat(replay_parts, ignore_index=True)
    common = pd.concat(common_parts, ignore_index=True)
    assert len(replay) == 729 * 58 and replay.seed_count.eq(5).all()
    replay.to_csv(OUT / 'teacher_predictions_private.csv', index=False, float_format='%.17g')
    common.to_csv(OUT / 'common_features_private.csv', index=False, float_format='%.17g')
    test_check = source.merge(replay, on=KEYS, validate='one_to_one', suffixes=('_old', '_replay'))
    assert len(test_check) == 5184
    replay_difference = float(np.max(np.abs(test_check.model_probability_AD_old - test_check.model_probability_AD_replay)))
    assert replay_difference < 1e-6, replay_difference
    historical = pd.read_csv(OLD / 'development_predictions_private.csv', dtype={'_RID': str})
    predictions, records, fits = [], [], {}
    reference_replay_difference = 0.
    for (family, config, domain), evaluation in source.groupby(['family', 'configuration_id', 'outer_domain'], sort=True):
        training = source[source.family.eq(family) & source.configuration_id.eq(config) & source.outer_domain.ne(domain)]
        assert set(training._RID).isdisjoint(evaluation._RID)
        common_domain = common[common.teacher_domain.eq(domain)].drop(columns=['teacher_domain'])
        train = training[['_RID', 'model_logit']].merge(common_domain, on='_RID', validate='one_to_one')
        val = evaluation[KEYS + ['model_logit', 'pathology_probability_AD']].merge(common_domain, on='_RID', validate='one_to_one')
        assert train.actual_domain.ne(domain).all() and val.actual_domain.eq(domain).all()
        qdiff = float(np.max(np.abs(val.reference_logit - logit(val.pathology_probability_AD))))
        reference_replay_difference = max(reference_replay_difference, qdiff)
        assert qdiff < 1e-8
        teacher = replay[replay.outer_domain.eq(domain) & replay.family.eq(family) & replay.configuration_id.eq(config)]
        train = train.merge(teacher[['_RID', 'teacher_logit']], on='_RID', validate='one_to_one')
        block = val[KEYS + ['model_logit']].copy()
        old = historical[historical.family.eq(family) & historical.configuration_id.eq(config) & historical.outer_domain.eq(domain)]
        block = block.merge(old[KEYS + ['prediction_AD4_LINEAR']], on=KEYS, validate='one_to_one').rename(columns={'prediction_AD4_LINEAR': 'prediction_HISTORICAL_AD4'})
        for name in ['MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']:
            cols = ['reference_logit'] if name == 'MATCHED_REF1' else AD_COLS
            target = 'model_logit' if name == 'MIXED_AD4' else 'teacher_logit'
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.))
            model.fit(train[cols].to_numpy(float), train[target].to_numpy(float))
            block['prediction_' + name] = model.predict(val[cols].to_numpy(float))
            fits[(family, config, domain, name)] = model
            records.append({'family': family, 'configuration_id': config, 'outer_domain': domain,
                            'method': name, 'training_people': len(train), 'evaluation_people': len(val),
                            'training_domains': sorted(train.actual_domain.unique()),
                            'features': cols, 'target': target, 'alpha': 1.,
                            'teacher_minus_mixed_target_mae': float(np.abs(train.teacher_logit - train.model_logit).mean())})
        predictions.append(block)
    result_rows = pd.concat(predictions, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    mae, contrasts, groups, boot = summarize(result_rows)
    result_rows.to_csv(OUT / 'development_predictions_private.csv', index=False, float_format='%.17g')
    groups.to_csv(OUT / 'group_results_aggregate.csv', index=False)
    np.save(OUT / 'participant_contrasts_bootstrap.npy', boot)
    joblib.dump(fits, OUT / 'source_fitted_models.joblib')
    (OUT / 'fit_records.json').write_text(json.dumps(records, indent=2) + '\n')
    (OUT / 'teacher_split_checks.json').write_text(json.dumps(split_receipts, indent=2) + '\n')
    checkpoint_hashes = {}
    for job in manifest.fw5_job_id:
        directory = JOBS_ROOT / job
        for filename in ['checkpoint.pt', 'preprocessor.npz', 'summary.json', 'adjacency.npy']:
            p = directory / filename
            if p.exists():
                checkpoint_hashes[str(p.relative_to(ROOT))] = sha(p)
    (OUT / 'teacher_artifact_hashes.json').write_text(json.dumps(checkpoint_hashes, indent=2) + '\n')
    pass_gate = all(c['gate_pass'] for c in contrasts.values())
    outputs = ['teacher_predictions_private.csv', 'common_features_private.csv', 'development_predictions_private.csv',
               'group_results_aggregate.csv', 'participant_contrasts_bootstrap.npy', 'source_fitted_models.joblib',
               'fit_records.json', 'teacher_split_checks.json', 'teacher_artifact_hashes.json']
    summary = {
        'status': 'SOURCE_GATE_PASSED_EXPLORATORY_PAIR_STAGE_ELIGIBLE' if pass_gate else 'SOURCE_GATE_FAILED_STOP_CANDIDATE',
        'people': 650, 'prediction_rows': len(result_rows), 'configuration_domain_groups': len(predictions),
        'source_teacher_prediction_rows': len(replay), 'mae_logit': mae, 'primary_contrasts': contrasts,
        'replay_probability_max_difference': replay_difference, 'reference_logit_max_difference': reference_replay_difference,
        'source_sha256': provenance, 'output_sha256': {n: sha(OUT / n) for n in outputs},
        'confirmation_or_pair_evaluated': False, 'new_mri_training': False,
        'limits': ['Previously examined development cohort, exploratory not independent confirmation',
                   'Teacher trained on surrogate-source participants; valid for distilling its behavior, not an honest source error-risk estimate',
                   'Intervals conditional on all fitted source objects; no fit uncertainty',
                   'No error-review, clinical, causal or reader utility result'],
        'runtime_seconds': time.perf_counter() - started,
    }
    (OUT / 'development_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['status', 'people', 'prediction_rows', 'mae_logit', 'primary_contrasts', 'runtime_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
