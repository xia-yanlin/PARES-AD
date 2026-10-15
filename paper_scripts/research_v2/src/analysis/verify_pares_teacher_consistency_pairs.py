"""Verify frozen-fit reuse and natural-pair arithmetic without evaluator imports."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research_v2/outputs/private/pares_ad_teacher_consistency_2026-10-09'
OLD = ROOT / 'research_v2/outputs/private/pares_ad_observed_pairs_2026-10-09'
METHODS = ['OLD_AD4', 'OLD_REF1', 'MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']


def main():
    result = json.loads((OUT / 'pair_summary.json').read_text())
    source = json.loads((OUT / 'development_summary.json').read_text())
    for name, digest in result['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in result['output_sha256'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUT / 'source_fitted_models.joblib').read_bytes()).hexdigest() == source['output_sha256']['source_fitted_models.joblib']
    old = json.loads((OLD / 'evaluation.json').read_text())
    for name in ['data\\raw\\tabular\\ADSP_PHC_T1_FS_06Jul2026.csv', 'research_v2\\outputs\\private\\pares_ad_observed_pairs_2026-10-09\\pair_manifest_private.csv']:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == old['source_sha256'][name]
    manifest = pd.read_csv(OLD / 'pair_manifest_private.csv', dtype={'_RID': str})
    estimates = pd.read_csv(OUT / 'pair_scan_estimates_private.csv', dtype={'_RID': str, 'participant': str})
    features = pd.read_csv(OUT / 'pair_features_private.csv', dtype={'_RID': str, 'participant': str})
    prior = pd.read_csv(OLD / 'scan_predictions_private.csv', dtype={'_RID': str, 'participant': str})
    key = ['_RID', 'participant', 'visit', 'outer_domain', 'family', 'configuration_id']
    joint = estimates.merge(prior[key + ['model_probability_AD', 'estimate_AD4', 'estimate_REF1']], on=key, suffixes=('', '_original'), validate='one_to_one')
    assert len(joint) == 6538 and set(joint.participant) == set(manifest._RID)
    assert len(features) == 804 and not features.duplicated(['_RID', 'outer_domain']).any()
    assert not estimates.duplicated(key).any()
    probability_roundtrip_difference = float(np.max(np.abs(joint.model_probability_AD - joint.model_probability_AD_original)))
    assert probability_roundtrip_difference < 1e-15
    assert np.allclose(joint.prediction_OLD_AD4, joint.estimate_AD4_original, atol=1e-12)
    assert np.allclose(joint.prediction_OLD_REF1, joint.estimate_REF1_original, atol=1e-12)
    p = np.clip(joint.model_probability_AD.to_numpy(), 1e-4, 1 - 1e-4)
    joint['independent_logit'] = np.log(p) - np.log1p(-p)
    assert np.allclose(joint.independent_logit, joint.model_logit, atol=1e-10)
    fits = joblib.load(OUT / 'source_fitted_models.joblib')
    max_prediction_difference = 0.
    for (f, c, d), block in joint.groupby(['family', 'configuration_id', 'outer_domain']):
        for name in ['MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']:
            cols = ['reference_logit'] if name == 'MATCHED_REF1' else [f'C00{i}_severity' for i in range(1, 5)]
            model = fits[(f, c, d, name)]
            manual = (block[cols].to_numpy() - model[0].mean_) / model[0].scale_ @ model[1].coef_ + model[1].intercept_
            gap = float(np.max(np.abs(manual - block['prediction_' + name])))
            assert gap < 1e-10
            max_prediction_difference = max(max_prediction_difference, gap)
    pair_key = ['participant', 'outer_domain', 'family', 'configuration_id']
    vals = ['independent_logit'] + ['prediction_' + m for m in METHODS]
    pairs = joint[joint.visit.eq('anchor')][pair_key + vals].merge(
        joint[joint.visit.eq('followup')][pair_key + vals], on=pair_key, suffixes=('_a', '_b'), validate='one_to_one')
    assert len(pairs) == 3269 and pairs.participant.nunique() == 402
    observed = pairs.independent_logit_b - pairs.independent_logit_a
    for m in METHODS:
        pairs['loss_' + m] = (pairs['prediction_' + m + '_b'] - pairs['prediction_' + m + '_a'] - observed).abs()
    people = pairs.groupby('participant', sort=True)[['loss_' + m for m in METHODS]].mean()
    for m in METHODS:
        assert abs(people['loss_' + m].mean() - result['mae_logit_change'][m]) < 1e-10
    rng = np.random.default_rng(20261022)
    comparisons = ['OLD_AD4', 'MATCHED_REF1']
    boot = []
    for _ in range(2000):
        sample = people.iloc[rng.choice(402, 402, replace=True)]
        boot.append([float((sample['loss_' + m] - sample.loss_MATCHED_AD4).mean()) for m in comparisons])
    bootstrap_difference = float(np.max(np.abs(np.asarray(boot) - np.load(OUT / 'pair_contrasts_bootstrap.npy'))))
    assert bootstrap_difference < 1e-10
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    gates = []
    for i, m in enumerate(comparisons):
        record = result['primary_contrasts'][m]
        assert np.allclose(intervals[:, i], record['ci95'], atol=1e-10)
        config_losses = pairs.groupby('configuration_id')[['loss_' + m, 'loss_MATCHED_AD4']].mean()
        directions = int((config_losses['loss_' + m] > config_losses.loss_MATCHED_AD4).sum())
        assert directions == record['configurations_improved']
        gain = people['loss_' + m].mean() - people.loss_MATCHED_AD4.mean()
        assert abs(gain - record['absolute_mae_reduction']) < 1e-10
        gate = gain / people['loss_' + m].mean() >= .1 and intervals[0, i] > 0 and directions >= 7
        assert bool(gate) == record['gate_pass']
        gates.append(gate)
    assert (result['status'] == 'EXPLORATORY_PAIR_GATE_FAILED') == (not all(gates))
    verification = {'status': 'PASS', 'people': 402, 'configuration_pairs': 3269,
                    'source_surrogate_unchanged': True, 'original_pairs_and_probabilities_unchanged': True,
                    'csv_probability_roundtrip_max_difference': probability_roundtrip_difference,
                    'saved_linear_prediction_max_difference': max_prediction_difference,
                    'participant_bootstrap_max_difference': bootstrap_difference,
                    'scope': 'Checks frozen-fit application, full prior case retention and paired statistics; no independent biological or human-use validation.'}
    (OUT / 'pair_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
