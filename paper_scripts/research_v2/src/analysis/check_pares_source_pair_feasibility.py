"""Inspect fixed source anchors and observed followups without change outcomes."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from research_v2.src.analysis.confirm_pares_ad_independent import (
    RAW_FS, BASELINE_CONCEPTS, MAPPING, canonical_rid,
    canonicalize_phase, canonicalize_manufacturer, canonicalize_field_strength,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'research_v2/protocols/pares_ad_source_pair_feasibility_2026-10-09.json'
OUT = ROOT / 'research_v2/outputs/private/pares_ad_source_pair_development_2026-10-09'
SOURCE = ROOT / 'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08/source_full_private.csv'
CONFIRMATION = ROOT / 'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08/adni_scores_private.csv'
CONFIRMATION_PAIRS = ROOT / 'research_v2/outputs/private/pares_ad_observed_pairs_2026-10-09/pair_manifest_private.csv'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def annotate(frame):
    f = frame.copy()
    f['_RID'] = canonical_rid(f.RID)
    f['date'] = pd.to_datetime(f.PHC_SCANDATE, errors='coerce')
    f['image_id'] = f.PHC_LONIUID.fillna('').astype(str).str.strip()
    f['phase'] = f.PHASE.map(canonicalize_phase)
    f['manufacturer'] = f.ScannerManufacturer.map(canonicalize_manufacturer)
    f['field_strength'] = f.FieldSgth.map(canonicalize_field_strength)
    f['outer_domain'] = f.phase + '_' + f.manufacturer
    return f


def main():
    if (OUT / 'feasibility.json').exists():
        raise RuntimeError('Feasibility already exists; preserve it.')
    cfg = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    mapping = json.loads(MAPPING.read_text())['nodes']
    required = ['EstimatedTotalIntraCranialVol'] + [n['volume_col'] for n in mapping] + [n['thickness_col'] for n in mapping if n['thickness_col'] is not None]
    assert len(required) == 163
    source = pd.read_csv(SOURCE, dtype={'_RID': str}, usecols=['_RID', 'outer_domain']).drop_duplicates()
    assert len(source) == 650 and not source._RID.duplicated().any()
    baseline = pd.read_csv(BASELINE_CONCEPTS, low_memory=False)
    baseline = baseline[canonical_rid(baseline.RID).isin(source._RID)]
    anchor = annotate(baseline)
    raw = pd.read_csv(RAW_FS, low_memory=False)
    raw['raw_row'] = np.arange(len(raw))
    raw = raw[canonical_rid(raw.RID).isin(source._RID)]
    raw = raw[raw.PHASE.fillna('').astype(str).str.strip().str.upper().isin(['ADNI1', 'ADNI3', 'ADNI4'])]
    raw = annotate(raw)
    vals = raw[required].apply(pd.to_numeric, errors='coerce')
    raw['valid_inputs'] = np.isfinite(vals).all(axis=1) & vals.gt(0).all(axis=1)
    metadata = ['phase', 'manufacturer', 'field_strength', 'ScannerBatchName']
    flow = {'source_anchors': len(anchor), 'unresolved_anchor_identity': 0, 'invalid_anchor_inputs': 0,
            'with_time_window': 0, 'with_metadata_match': 0, 'with_valid_followup_inputs': 0,
            'best_distance_ties': 0}
    pairs = []
    for a in anchor.to_dict('records'):
        same = raw[raw._RID.eq(a['_RID']) & raw.date.eq(a['date']) & raw.image_id.eq(a['image_id'])]
        identity_valid = len(same) == 1 and a['image_id'] != ''
        if identity_valid:
            record = same.iloc[0]
            identity_valid = all(pd.notna(a[c]) and str(a[c]).strip() != '' and a[c] == record[c] for c in metadata)
            identity_valid = identity_valid and np.allclose(np.asarray([a[c] for c in required], dtype=float),
                                                           record[required].to_numpy(float), atol=1e-10, rtol=0, equal_nan=True)
        if not identity_valid:
            flow['unresolved_anchor_identity'] += 1
            continue
        if not record.valid_inputs:
            flow['invalid_anchor_inputs'] += 1
            continue
        group = raw[raw._RID.eq(a['_RID'])].copy()
        group['days'] = (group.date - a['date']).dt.days
        group = group[group.days.between(180, 730)]
        if group.empty:
            continue
        flow['with_time_window'] += 1
        for c in metadata:
            group = group[group[c].notna() & group[c].eq(a[c])]
        if group.empty:
            continue
        flow['with_metadata_match'] += 1
        group = group[group.valid_inputs].copy()
        if group.empty:
            continue
        flow['with_valid_followup_inputs'] += 1
        group['distance'] = (group.days - 365).abs()
        best = group[group.distance.eq(group.distance.min())]
        if len(best) != 1:
            flow['best_distance_ties'] += 1
            continue
        b = best.iloc[0]
        pairs.append({'_RID': a['_RID'], 'outer_domain': a['outer_domain'], 'anchor_raw_row': int(record.raw_row),
                      'followup_raw_row': int(b.raw_row), 'days': int(b.days)})
    pair = pd.DataFrame(pairs, columns=['_RID', 'outer_domain', 'anchor_raw_row', 'followup_raw_row', 'days'])
    domains = sorted(source.outer_domain.unique())
    counts = pair.groupby('outer_domain').size().reindex(domains, fill_value=0)
    conf = set(pd.read_csv(CONFIRMATION, usecols=['_RID'], dtype=str)._RID)
    conf_pair = set(pd.read_csv(CONFIRMATION_PAIRS, usecols=['_RID'], dtype=str)._RID)
    assert set(source._RID).isdisjoint(conf) and set(pair._RID).isdisjoint(conf_pair)
    floors = cfg['operational_floors']
    outer_support = {d: {'other_domain_pairs': int(len(pair) - counts[d]),
                         'other_domains_with10': int(counts.drop(d).ge(10).sum())} for d in domains}
    passed = len(pair) >= floors['minimum_total_pairs'] and counts.ge(20).sum() >= floors['minimum_domains_with20']
    passed = passed and flow['unresolved_anchor_identity'] == 0
    passed = passed and all(v['other_domain_pairs'] >= floors['each_target_domain_min_other_domain_pairs'] and
                            v['other_domains_with10'] >= floors['each_target_domain_min_other_domains_with10'] for v in outer_support.values())
    OUT.mkdir(parents=True, exist_ok=True)
    pair.to_csv(OUT / 'pair_manifest_private.csv', index=False)
    result = {'status': 'FEASIBLE_FOR_FROZEN_METHOD_PROTOCOL' if passed else 'STOP_SOURCE_PAIR_FEASIBILITY',
              'flow': flow, 'eligible_people': len(pair), 'people_by_domain': {k: int(v) for k, v in counts.items()},
              'outer_training_support': outer_support, 'confirmation_participant_overlap': 0,
              'interval_days': {'min': int(pair.days.min()), 'median': float(pair.days.median()), 'max': int(pair.days.max())} if len(pair) else None,
              'new_followup_inference': False, 'roi_changes_inspected': False,
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [PROTOCOL, Path(__file__), RAW_FS, BASELINE_CONCEPTS, MAPPING, SOURCE, CONFIRMATION, CONFIRMATION_PAIRS]},
              'pair_manifest_sha256': sha(OUT / 'pair_manifest_private.csv'),
              'scope': 'Source method-development feasibility, not power or efficacy; metadata match does not prove physical scanner identity.'}
    (OUT / 'feasibility.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256', 'pair_manifest_sha256']}, indent=2))


if __name__ == '__main__':
    main()
