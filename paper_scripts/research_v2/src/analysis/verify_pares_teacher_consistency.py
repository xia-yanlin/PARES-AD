"""Independent source-key, split, closed-form and endpoint verification."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research_v2/outputs/private/pares_ad_teacher_consistency_2026-10-09'
SOURCE = ROOT / 'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08/source_full_private.csv'
MEMBERSHIP = ROOT / 'research_v2/outputs/private/nested_domain_splits.csv'
KEYS = ['_RID', 'outer_domain', 'family', 'configuration_id']
METHODS = ['HISTORICAL_AD4', 'MIXED_AD4', 'MATCHED_AD4', 'MATCHED_REF1']
COMPARATORS = ['MIXED_AD4', 'MATCHED_REF1', 'HISTORICAL_AD4']


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), .0001, .9999)
    return np.log(p) - np.log1p(-p)


def main():
    summary = json.loads((OUT / 'development_summary.json').read_text())
    for name, h in summary['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h
    for name, h in summary['output_sha256'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == h
    for name, h in json.loads((OUT / 'teacher_artifact_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h
    source = pd.read_csv(SOURCE, dtype={'_RID': str})
    source['original_logit'] = to_logit(source.model_probability_AD)
    predictions = pd.read_csv(OUT / 'development_predictions_private.csv', dtype={'_RID': str})
    teacher = pd.read_csv(OUT / 'teacher_predictions_private.csv', dtype={'_RID': str})
    common = pd.read_csv(OUT / 'common_features_private.csv', dtype={'_RID': str})
    membership = pd.read_csv(MEMBERSHIP, dtype={'RID': str})
    records = json.loads((OUT / 'fit_records.json').read_text())
    fitted = joblib.load(OUT / 'source_fitted_models.joblib')
    joint = source.merge(predictions, on=KEYS, validate='one_to_one')
    assert len(joint) == 5184 and joint._RID.nunique() == 650
    assert np.allclose(joint.original_logit, joint.model_logit, atol=1e-10)
    assert teacher.seed_count.eq(5).all() and len(teacher) == 729 * 58
    assert not teacher.duplicated(KEYS).any()
    assert not common.duplicated(['_RID', 'teacher_domain']).any()
    replay = source.merge(teacher, on=KEYS, validate='one_to_one', suffixes=('_source', '_teacher'))
    replay_difference = float(np.max(np.abs(replay.model_probability_AD_source - replay.model_probability_AD_teacher)))
    assert replay_difference < 1e-6
    assert abs(replay_difference - summary['replay_probability_max_difference']) < 1e-12
    assert np.allclose(to_logit(teacher.model_probability_AD), teacher.teacher_logit, atol=1e-10)
    fit_error = 0.
    for row in records:
        f, c, d, m = (row[k] for k in ['family', 'configuration_id', 'outer_domain', 'method'])
        train_keys = source[source.family.eq(f) & source.configuration_id.eq(c) & source.outer_domain.ne(d)]
        val_keys = joint[joint.family.eq(f) & joint.configuration_id.eq(c) & joint.outer_domain.eq(d)]
        assert set(train_keys._RID).isdisjoint(val_keys._RID)
        actual_train_ids = set(membership.loc[membership.outer_domain.eq(d) & membership.outer_role.eq('train_pool'), 'RID'])
        actual_test_ids = set(membership.loc[membership.outer_domain.eq(d) & membership.outer_role.eq('test'), 'RID'])
        assert actual_train_ids.isdisjoint(actual_test_ids)
        assert set(val_keys._RID) <= actual_test_ids
        assert set(train_keys._RID) <= actual_train_ids
        ft = common[common.teacher_domain.eq(d)]
        train = train_keys[['_RID', 'original_logit']].merge(ft, on='_RID', validate='one_to_one')
        val = val_keys.merge(ft, on='_RID', validate='one_to_one')
        assert train.actual_domain.ne(d).all() and val.actual_domain.eq(d).all()
        assert sorted(train.actual_domain.unique()) == row['training_domains']
        exact_teacher = teacher[teacher.family.eq(f) & teacher.configuration_id.eq(c) & teacher.outer_domain.eq(d)]
        train = train.merge(exact_teacher[['_RID', 'teacher_logit']], on='_RID', validate='one_to_one')
        y = train.original_logit.to_numpy() if m == 'MIXED_AD4' else train.teacher_logit.to_numpy()
        x, v = train[row['features']].to_numpy(), val[row['features']].to_numpy()
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd == 0, 1., sd)
        z = (x - mu) / sd
        beta = np.linalg.solve(z.T @ z + np.eye(x.shape[1]), z.T @ (y - y.mean()))
        manual = y.mean() + ((v - mu) / sd) @ beta
        model = fitted[(f, c, d, m)]
        assert np.allclose(mu, model[0].mean_, atol=1e-10)
        assert np.allclose(sd, model[0].scale_, atol=1e-10)
        gap = float(np.max(np.abs(manual - val['prediction_' + m])))
        fit_error = max(fit_error, gap)
        assert gap < 1e-8
    losses = {}
    for m in METHODS:
        loss = (joint['prediction_' + m] - joint.original_logit).abs()
        assert np.allclose(loss, joint['loss_' + m], atol=1e-10)
        losses[m] = pd.Series(loss.to_numpy(), index=joint._RID).groupby(level=0, sort=True).mean().to_numpy()
        assert abs(losses[m].mean() - summary['mae_logit'][m]) < 1e-10
    rng = np.random.default_rng(20261022)
    bootstrap = []
    for _ in range(2000):
        indices = rng.choice(650, 650, replace=True)
        bootstrap.append([float(np.mean(losses[c][indices] - losses['MATCHED_AD4'][indices])) for c in COMPARATORS])
    bootstrap_difference = float(np.max(np.abs(np.asarray(bootstrap) - np.load(OUT / 'participant_contrasts_bootstrap.npy'))))
    assert bootstrap_difference < 1e-10
    cis = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    gates = []
    for i, c in enumerate(COMPARATORS):
        row = summary['primary_contrasts'][c]
        assert np.allclose(cis[:, i], row['ci95'], atol=1e-10)
        delta = losses[c].mean() - losses['MATCHED_AD4'].mean()
        relative = delta / losses[c].mean()
        directions = 0
        for _, group in joint.groupby('configuration_id'):
            directions += bool(group['loss_' + c].mean() > group.loss_MATCHED_AD4.mean())
        assert directions == row['configurations_improved']
        assert abs(relative - row['relative_reduction']) < 1e-10
        gate = relative >= .1 and cis[0, i] > 0 and directions >= 7
        assert bool(gate) == row['gate_pass']
        gates.append(gate)
    assert (summary['status'] == 'SOURCE_GATE_FAILED_STOP_CANDIDATE') == (not all(gates))
    result = {'status': 'PASS', 'verified_fits': len(records), 'people': 650,
              'probability_replay_max_difference': replay_difference,
              'closed_form_prediction_max_difference': fit_error,
              'bootstrap_max_difference': bootstrap_difference,
              'split_membership_checked_from_original_file': True,
              'teacher_artifacts_unchanged': True,
              'scope': 'Checks original split keys, fixed teacher replay, saved common features, linear fits and endpoints; not an independent implementation of concept transforms or a clinical validity result.'}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
