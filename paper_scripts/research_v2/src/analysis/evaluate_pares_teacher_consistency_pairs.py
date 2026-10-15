"""Apply frozen teacher-matched source surrogates to existing natural pairs."""
from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from research_v2.src.analysis.confirm_pares_ad_independent import RAW_FS, BASELINE_CONCEPTS, ASSIGNMENTS
from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS, CONCEPT_ORIENTATION, fit_monotone_pathology_model, logit,
)
from research_v2.src.audit.concepts import build_concept_target

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research_v2/outputs/private/pares_ad_teacher_consistency_2026-10-09'
PAIRS = ROOT / 'research_v2/outputs/private/pares_ad_observed_pairs_2026-10-09'
METHODS = ['OLD_AD4', 'OLD_REF1', 'MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']
COMPARATORS = ['OLD_AD4', 'MATCHED_REF1']
AD_COLS = [f'{c}_severity' for c in CORE_CONCEPTS]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / 'pairs_started.json').exists():
        raise RuntimeError('Pair stage already started; inspect its state before retrying.')
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    summary = json.loads((OUT / 'development_summary.json').read_text())
    assert summary['status'] == 'SOURCE_GATE_PASSED_EXPLORATORY_PAIR_STAGE_ELIGIBLE'
    assert json.loads((OUT / 'verification.json').read_text())['status'] == 'PASS'
    assert sha(OUT / 'source_fitted_models.joblib') == summary['output_sha256']['source_fitted_models.joblib']
    protocol = ROOT / 'research_v2/protocols/pares_ad_teacher_consistency_2026-10-09.json'
    assert sha(protocol) == summary['source_sha256'][str(protocol.relative_to(ROOT))]
    # The already-fitted normalization/reference was saved as source values,
    # not a serialized transformer. Replay its identical source-only recipe,
    # checking all 729 frozen values per domain before applying it to scans.
    # No surrogate, teacher, hyperparameter or evaluation-dependent fit changes.
    execution_note = {
        'protocol_sha256': sha(protocol), 'script_sha256': sha(Path(__file__)),
        'source_surrogate_sha256': sha(OUT / 'source_fitted_models.joblib'),
        'preprocessing_replay': 'Deterministic replay from identical frozen baseline outside target domain; must reproduce every saved common source feature and reference value before use. No new surrogate/teacher fitting or tuning.',
        'new_scan_predictions': False,
    }
    (OUT / 'pairs_started.json').write_text(json.dumps(execution_note, indent=2) + '\n')
    manifest = pd.read_csv(PAIRS / 'pair_manifest_private.csv', dtype={'_RID': str})
    baseline = pd.read_csv(BASELINE_CONCEPTS, dtype={'RID': str}, low_memory=False)
    assignment = pd.read_csv(ASSIGNMENTS, dtype={'RID': str})
    baseline = baseline.merge(assignment[['RID', 'acquisition_domain']], on='RID', validate='one_to_one')
    baseline['_RID'] = baseline.RID
    raw = pd.read_csv(RAW_FS, low_memory=False)
    scans = []
    for visit, index_name in [('anchor', 'anchor_raw_row'), ('followup', 'followup_raw_row')]:
        frame = raw.iloc[manifest[index_name].to_numpy(int)].copy().reset_index(drop=True)
        assert np.array_equal(pd.to_numeric(frame.RID).astype('Int64').astype(str), manifest._RID)
        frame['_RID'] = [f'pair{i:04d}_{visit}' for i in range(len(manifest))]
        frame['participant'] = manifest._RID.to_numpy()
        frame['outer_domain'] = manifest.outer_domain.to_numpy()
        frame['visit'] = visit
        scans.append(frame)
    scans = pd.concat(scans, ignore_index=True)
    assert set(scans.participant).isdisjoint(baseline._RID)
    frozen_common = pd.read_csv(OUT / 'common_features_private.csv', dtype={'_RID': str})
    transformed = []
    max_replay = 0.
    for domain, evaluation in scans.groupby('outer_domain', sort=True):
        train = baseline[baseline.acquisition_domain.ne(domain)].reset_index(drop=True)
        evaluation = evaluation.reset_index(drop=True)
        # The baseline prefix allows exact replay verification; it is never
        # treated as new outcome data for the stage-two comparison.
        all_eval = pd.concat([baseline, evaluation], ignore_index=True)
        x_train, x_eval = [], []
        for c in CORE_CONCEPTS:
            target = build_concept_target(c, train, all_eval)
            assert target.train_valid.all() and target.evaluation_valid.all()
            x_train.append(CONCEPT_ORIENTATION[c] * target.train_values)
            x_eval.append(CONCEPT_ORIENTATION[c] * target.evaluation_values)
        x_train, x_eval = np.column_stack(x_train), np.column_stack(x_eval)
        intercept, beta = fit_monotone_pathology_model(x_train, train.AD_label.to_numpy(int), .1)
        reference = logit(expit(intercept + x_eval @ beta))
        expected = baseline[['_RID']].merge(frozen_common[frozen_common.teacher_domain.eq(domain)], on='_RID', validate='one_to_one')
        error = max(float(np.max(np.abs(x_eval[:len(baseline)] - expected[AD_COLS].to_numpy()))),
                    float(np.max(np.abs(reference[:len(baseline)] - expected.reference_logit))))
        max_replay = max(max_replay, error)
        assert error < 1e-9
        block = evaluation[['_RID', 'participant', 'visit', 'outer_domain']].copy()
        block[AD_COLS] = x_eval[len(baseline):]
        block['reference_logit'] = reference[len(baseline):]
        transformed.append(block)
    transformed = pd.concat(transformed, ignore_index=True)
    transformed.to_csv(OUT / 'pair_features_private.csv', index=False, float_format='%.17g')
    fits = joblib.load(OUT / 'source_fitted_models.joblib')
    prior = pd.read_csv(PAIRS / 'scan_predictions_private.csv', dtype={'participant': str, '_RID': str})
    joint = prior.merge(transformed, on=['_RID', 'participant', 'visit', 'outer_domain'], validate='many_to_one')
    assert len(joint) == 6538
    joint['prediction_OLD_AD4'] = joint.estimate_AD4
    joint['prediction_OLD_REF1'] = joint.estimate_REF1
    for (family, config, domain), indices in joint.groupby(['family', 'configuration_id', 'outer_domain'], sort=True).groups.items():
        frame = joint.loc[indices]
        for m in ['MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']:
            cols = ['reference_logit'] if m == 'MATCHED_REF1' else AD_COLS
            joint.loc[indices, 'prediction_' + m] = fits[(family, config, domain, m)].predict(frame[cols].to_numpy(float))
    assert np.isfinite(joint[['prediction_' + m for m in METHODS]]).all().all()
    keys = ['participant', 'outer_domain', 'family', 'configuration_id']
    values = ['model_logit'] + ['prediction_' + m for m in METHODS]
    wide = joint[joint.visit.eq('anchor')][keys + values].merge(
        joint[joint.visit.eq('followup')][keys + values], on=keys, suffixes=('_anchor', '_followup'), validate='one_to_one')
    assert len(wide) == 3269 and wide.participant.nunique() == 402
    wide['observed_delta'] = wide.model_logit_followup - wide.model_logit_anchor
    for m in METHODS:
        wide['delta_' + m] = wide['prediction_' + m + '_followup'] - wide['prediction_' + m + '_anchor']
        wide['loss_' + m] = (wide['delta_' + m] - wide.observed_delta).abs()
    losses = ['loss_' + m for m in METHODS]
    people = wide.groupby('participant', sort=True)[losses].mean()
    mae = {m: float(people['loss_' + m].mean()) for m in METHODS}
    old_summary = json.loads((PAIRS / 'evaluation.json').read_text())
    assert abs(mae['OLD_AD4'] - old_summary['mae']['AD4']) < 1e-10
    assert abs(mae['OLD_REF1'] - old_summary['mae']['REF1']) < 1e-10
    aggregates = []
    for scope, field in [('configuration', 'configuration_id'), ('domain', 'outer_domain')]:
        for value, group in wide.groupby(field):
            mean = group.groupby('participant')[losses].mean().mean()
            aggregates.append({'scope': scope, 'value': value, 'people': group.participant.nunique(),
                               **{'mae_' + m: float(mean['loss_' + m]) for m in METHODS}})
    aggregate = pd.DataFrame(aggregates)
    configs = aggregate[aggregate.scope.eq('configuration')]
    rng = np.random.default_rng(20261022)
    contrast_array = np.column_stack([people['loss_' + m] - people.loss_MATCHED_AD4 for m in COMPARATORS])
    boot = np.empty((2000, 2))
    for i in range(2000):
        boot[i] = contrast_array[rng.integers(0, len(people), len(people))].mean(axis=0)
    intervals = np.quantile(boot, [.025, .975], axis=0)
    contrasts = {}
    for i, m in enumerate(COMPARATORS):
        gain = mae[m] - mae['MATCHED_AD4']
        relative = gain / mae[m]
        direction = int((configs.mae_MATCHED_AD4 < configs['mae_' + m]).sum())
        contrasts[m] = {'absolute_mae_reduction': gain, 'relative_reduction': relative,
                        'ci95': intervals[:, i].tolist(), 'configurations_improved': direction,
                        'gate_pass': bool(relative >= .1 and intervals[0, i] > 0 and direction >= 7)}
    joint.to_csv(OUT / 'pair_scan_estimates_private.csv', index=False, float_format='%.17g')
    wide.to_csv(OUT / 'pair_results_private.csv', index=False, float_format='%.17g')
    aggregate.to_csv(OUT / 'pair_groups_aggregate.csv', index=False)
    np.save(OUT / 'pair_contrasts_bootstrap.npy', boot)
    outputs = ['pair_features_private.csv', 'pair_scan_estimates_private.csv', 'pair_results_private.csv', 'pair_groups_aggregate.csv', 'pair_contrasts_bootstrap.npy']
    result = {
        'status': 'EXPLORATORY_PAIR_GATE_PASSED' if all(c['gate_pass'] for c in contrasts.values()) else 'EXPLORATORY_PAIR_GATE_FAILED',
        'people': 402, 'configuration_pairs': 3269, 'mae_logit_change': mae, 'primary_contrasts': contrasts,
        'source_transform_replay_max_difference': max_replay,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), protocol, RAW_FS, BASELINE_CONCEPTS, ASSIGNMENTS, PAIRS / 'pair_manifest_private.csv', PAIRS / 'scan_predictions_private.csv', OUT / 'source_fitted_models.joblib']},
        'output_sha256': {n: sha(OUT / n) for n in outputs},
        'limitations': ['Previously used participants and previously known pair outcomes; exploratory reuse',
                        'No new MRI model fit, scan inference, surrogate fit or hyperparameter selection',
                        'Source feature/reference transforms deterministically replayed and checked against all frozen source values',
                        'Conditional fixed-fit intervals; no clinical, causal, review-utility or independent-validation claim'],
    }
    (OUT / 'pair_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'people', 'configuration_pairs', 'mae_logit_change', 'primary_contrasts']}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
