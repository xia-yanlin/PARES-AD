"""Replay frozen ADNI development models with the NACC unavailable-input mask."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from research_v2.src.analysis.confirm_pares_ad_independent import (
    BASELINE_CONCEPTS, ASSIGNMENTS, MAPPING, REGISTRY, MANIFEST, run_black_boxes, sha256_file
)

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research_v2/outputs/private/nacc_external_validation_2026-10-07/source_input_bridge'
PROTOCOL=ROOT/'research_v2/protocols/nacc_source_input_bridge_freeze_2026-10-07.md'
PREDICTIONS=ROOT/'research_v2/outputs/private/fw5/aggregation/fw5_outer_endpoints_v2_1/configuration_outer_predictions.csv'
KEY=['_RID','outer_domain','family','configuration_id']

def metric_group(group):
    y=group.AD_label.to_numpy()
    a=group.full.to_numpy(); b=group.masked.to_numpy()
    auc_a=float(roc_auc_score(y,a)) if len(np.unique(y))==2 else None
    auc_b=float(roc_auc_score(y,b)) if len(np.unique(y))==2 else None
    brier_a=float(brier_score_loss(y,a)); brier_b=float(brier_score_loss(y,b))
    auc_change=auc_b-auc_a if auc_a is not None else None
    brier_change=brier_b-brier_a
    return {'predictions':len(group), 'participants':int(group._RID.nunique()),
        'full_auroc':auc_a,'masked_auroc':auc_b,'delta_auroc':auc_change,
        'full_brier':brier_a,'masked_brier':brier_b,'delta_brier':brier_change,
        'mean_abs_probability_change':float(np.mean(np.abs(a-b))),
        'max_abs_probability_change':float(np.max(np.abs(a-b))),
        'class_change_fraction':float(np.mean((a>=.5)!=(b>=.5))),
        'materiality_flag':bool((auc_change is not None and auc_change < -.01) or brier_change > .01)}

def main():
    started=time.monotonic(); torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    mapping=json.loads(MAPPING.read_text())['nodes']
    registry=json.loads(REGISTRY.read_text())
    manifest=pd.read_csv(MANIFEST,dtype={'seed':int})
    original=pd.read_csv(PREDICTIONS,dtype={'RID':str}).rename(columns={'RID':'_RID'})
    baseline=pd.read_csv(BASELINE_CONCEPTS,dtype={'RID':str},low_memory=False)
    domains=pd.read_csv(ASSIGNMENTS,dtype={'RID':str})
    cohort=baseline.merge(domains[['RID','acquisition_domain']],on='RID',validate='one_to_one').rename(columns={'RID':'_RID','acquisition_domain':'outer_domain'})
    cohort=cohort.loc[cohort._RID.isin(original._RID)].copy()
    if len(cohort)!=650: raise RuntimeError('Unexpected source development receipt')
    receipt=json.loads((OUT.parent/'receipt/receipt_summary.json').read_text())
    missing=receipt['mapping']['absent_fields']
    print('Replaying all full-input development checkpoints.',flush=True)
    full=run_black_boxes(cohort,manifest,mapping,registry)
    reference=original[KEY+['mean_calibrated_probability_AD']]
    matched=full.merge(reference,on=KEY,validate='one_to_one',how='outer',indicator=True)
    if not matched['_merge'].eq('both').all(): raise RuntimeError('Development replay key mismatch')
    replay_diff=float(np.max(np.abs(matched.model_probability_AD-matched.mean_calibrated_probability_AD)))
    if replay_diff>1e-6: raise RuntimeError('Full-input replay probability tolerance failed')
    print(f'Full replay verified: max difference {replay_diff:.3g}. Replaying fixed missing-input mask.',flush=True)
    masked_cohort=cohort.copy(); masked_cohort[missing]=np.nan
    masked=run_black_boxes(masked_cohort,manifest,mapping,registry)
    paired=full[KEY+['AD_label','model_probability_AD']].rename(columns={'model_probability_AD':'full'}).merge(
        masked[KEY+['model_probability_AD']].rename(columns={'model_probability_AD':'masked'}),on=KEY,validate='one_to_one')
    rows=[{'scope':'overall','family':'all','configuration_id':'all',**metric_group(paired)}]
    for family,group in paired.groupby('family'):
        rows.append({'scope':'family','family':family,'configuration_id':'all',**metric_group(group)})
    for (family,config),group in paired.groupby(['family','configuration_id']):
        rows.append({'scope':'configuration','family':family,'configuration_id':config,**metric_group(group)})
    pd.DataFrame(rows).to_csv(OUT/'aggregate_comparison.csv',index=False)
    flag=any(row['materiality_flag'] for row in rows if row['scope'] in ('overall','configuration'))
    summary={'status':'MATERIAL_INPUT_CHANGE' if flag else 'NO_PRESPECIFIED_MATERIALITY_FLAG',
        'replay_max_absolute_difference':replay_diff,'source_development_participants':len(cohort),
        'source_model_case_predictions':len(paired),'missing_columns':missing,'rows':rows,
        'protocol_sha256':sha256_file(PROTOCOL),'script_sha256':sha256_file(Path(__file__)),
        'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in (BASELINE_CONCEPTS,ASSIGNMENTS,MAPPING,REGISTRY,MANIFEST,PREDICTIONS)},
        'nacc_predictions_computed':False,'adni_confirmation_used':False,
        'runtime_seconds':time.monotonic()-started}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'overall':rows[0],'runtime_seconds':summary['runtime_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
