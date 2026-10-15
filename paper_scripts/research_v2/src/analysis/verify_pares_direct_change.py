"""Independent arithmetic and split verification for direct-change development."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research_v2/outputs/private/pares_ad_source_pair_development_2026-10-09'
OLD = ROOT / 'research_v2/outputs/private/pares_ad_teacher_consistency_2026-10-09'
KEYS = ['_RID', 'outer_domain', 'family', 'configuration_id']
METHODS = ['LEVEL_AD4', 'DELTA_AD4', 'DELTA_REF1']


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), .0001, .9999)
    return np.log(p) - np.log1p(-p)


def main():
    summary = json.loads((OUT / 'development_summary.json').read_text())
    feasibility = json.loads((OUT / 'feasibility.json').read_text())
    for name, h in summary['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h
    for name, h in summary['output_sha256'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == h
    assert hashlib.sha256((OUT / 'pair_manifest_private.csv').read_bytes()).hexdigest() == feasibility['pair_manifest_sha256']
    pairs = pd.read_csv(OUT / 'pair_manifest_private.csv', dtype={'_RID': str})
    assert len(pairs) == 218 and not pairs._RID.duplicated().any()
    conf = pd.read_csv(ROOT / 'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08/adni_scores_private.csv', usecols=['_RID'], dtype=str)
    assert set(pairs._RID).isdisjoint(conf._RID)
    a = pd.read_csv(OLD / 'teacher_predictions_private.csv', dtype={'_RID': str})
    b = pd.read_csv(OUT / 'followup_teacher_predictions_private.csv', dtype={'_RID': str})
    assert a.seed_count.eq(5).all() and b.seed_count.eq(5).all()
    assert 'AD_label' not in b and len(b) == 218 * 58
    raw_delta = b.merge(a[KEYS + ['model_probability_AD']], on=KEYS, suffixes=('_b', '_a'), validate='one_to_one')
    raw_delta['manual_delta'] = logit(raw_delta.model_probability_AD_b) - logit(raw_delta.model_probability_AD_a)
    training = pd.read_csv(OUT / 'all_teacher_pair_deltas_private.csv', dtype={'_RID': str})
    raw_delta = raw_delta.merge(training[KEYS + ['observed_delta']], on=KEYS, validate='one_to_one')
    assert np.allclose(raw_delta.manual_delta, raw_delta.observed_delta, atol=1e-10)
    evaluated = pd.read_csv(OUT / 'development_predictions_private.csv', dtype={'_RID': str})
    expected = training[training.actual_domain.eq(training.outer_domain)]
    assert len(expected) == len(evaluated) == 1709
    assert set(evaluated._RID) == set(pairs._RID)
    assert set(map(tuple, expected[KEYS].to_numpy())) == set(map(tuple, evaluated[KEYS].to_numpy()))
    fits = joblib.load(OUT / 'source_delta_models.joblib')
    levels = joblib.load(OLD / 'source_fitted_models.joblib')
    records = json.loads((OUT / 'fit_records.json').read_text())
    membership = pd.read_csv(ROOT / 'research_v2/outputs/private/nested_domain_splits.csv', dtype={'RID': str})
    maximum_fit_difference = 0.
    for rec in records:
        f, c, d, m = (rec[n] for n in ['family', 'configuration_id', 'outer_domain', 'method'])
        group = training[training.family.eq(f) & training.configuration_id.eq(c) & training.outer_domain.eq(d)]
        train = group[group.actual_domain.ne(d)]
        val = evaluated[evaluated.family.eq(f) & evaluated.configuration_id.eq(c) & evaluated.outer_domain.eq(d)]
        assert set(train._RID).isdisjoint(val._RID)
        assert set(train._RID) <= set(membership.loc[membership.outer_domain.eq(d) & membership.outer_role.eq('train_pool'), 'RID'])
        assert set(val._RID) <= set(membership.loc[membership.outer_domain.eq(d) & membership.outer_role.eq('test'), 'RID'])
        assert len(train) == rec['training_people'] and len(val) == rec['evaluation_people']
        x, v = train[rec['features']].to_numpy(), val[rec['features']].to_numpy()
        scale = x.std(axis=0)
        scale = np.where(scale == 0, 1., scale)
        z = x / scale
        beta = np.linalg.solve(z.T @ z + np.eye(x.shape[1]), z.T @ train.observed_delta.to_numpy())
        manual = (v / scale) @ beta
        model = fits[(f, c, d, m)]
        assert not model[0].with_mean and not model[1].fit_intercept
        assert np.all(model.predict(np.zeros((1, len(rec['features'])))) == 0)
        gap = float(np.max(np.abs(manual - val['prediction_' + m])))
        assert gap < 1e-10
        maximum_fit_difference = max(maximum_fit_difference, gap)
        level = levels[(f, c, d, 'MATCHED_AD4')]
        dx = val[[f'delta_C00{i}_severity' for i in range(1, 5)]].to_numpy()
        assert np.allclose(dx / level[0].scale_ @ level[1].coef_, val.prediction_LEVEL_AD4, atol=1e-10)
    for method in METHODS:
        loss = (evaluated['prediction_' + method] - evaluated.observed_delta).abs()
        assert np.allclose(loss, evaluated['loss_' + method], atol=1e-10)
    people = evaluated.groupby('_RID', sort=True)[['loss_' + m for m in METHODS]].mean()
    for m in METHODS:
        assert abs(people['loss_' + m].mean() - summary['mae_logit_change'][m]) < 1e-10
    rng = np.random.default_rng(20261023)
    comparators = ['LEVEL_AD4', 'DELTA_REF1']
    boot = []
    for _ in range(2000):
        sample = people.iloc[rng.choice(218, 218, replace=True)]
        boot.append([float((sample['loss_' + m] - sample.loss_DELTA_AD4).mean()) for m in comparators])
    difference = float(np.max(np.abs(np.asarray(boot) - np.load(OUT / 'participant_contrasts_bootstrap.npy'))))
    assert difference < 1e-10
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    gates = []
    for i, m in enumerate(comparators):
        row = summary['primary_contrasts'][m]
        assert np.allclose(ci[:, i], row['ci95'], atol=1e-10)
        group = evaluated.groupby('configuration_id')[['loss_' + m, 'loss_DELTA_AD4']].mean()
        directions = int((group['loss_' + m] > group.loss_DELTA_AD4).sum())
        assert directions == row['configurations_improved']
        gain = people['loss_' + m].mean() - people.loss_DELTA_AD4.mean()
        relative = gain / people['loss_' + m].mean()
        assert abs(relative - row['relative_reduction']) < 1e-10
        passed = relative >= .1 and ci[0, i] > 0 and directions >= 7
        assert bool(passed) == row['gate_pass']
        gates.append(passed)
    assert (summary['status'] == 'SOURCE_GATE_FAILED_STOP_CANDIDATE') == (not all(gates))
    verification = {'status': 'PASS', 'people': 218, 'evaluated_configuration_pairs': 1709,
                    'verified_fits': len(records), 'max_closed_form_prediction_difference': maximum_fit_difference,
                    'max_participant_bootstrap_difference': difference,
                    'same_teacher_change_targets_checked': True, 'teacher_split_exclusion_checked': True,
                    'confirmation_subject_overlap': 0,
                    'scope': 'Original split memberships, frozen input provenance, no-intercept linear fits and change endpoints; not independent efficacy or clinical validity.'}
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
