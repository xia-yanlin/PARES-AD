"""Aggregate-only QC key compatibility diagnostic; no clinical outcomes read."""
from pathlib import Path
import json
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ZIP = ROOT / 'research_v2/data/received/nacc/request_19347_nacc75_202609/01_current_research_priority/investigator_scan_clariti_mri_csv_nacc75.zip'
OUT = ROOT / 'research_v2/outputs/private/nacc_external_validation_2026-10-07/receipt'
with zipfile.ZipFile(ZIP) as archive:
    with archive.open('investigator_scan_clariti_mrisbm_nacc75.csv') as s: m = pd.read_csv(s, dtype=str)
    with archive.open('investigator_scan_clariti_mriqc_nacc75.csv') as s: q = pd.read_csv(s, dtype=str)
result = {}
for source, frame, columns in [('analysis',m,['T1SERIESINSTANCEUID','LONIIMAGET1','SCANDT']),('qc',q,['SERIESINSTANCEUID','LONIIMAGE','STUDYDATE'])]:
    result[source] = {}
    for col in columns:
        vals = frame[col].fillna('').str.strip()
        result[source][col] = {'missing':int(vals.eq('').sum()), 'string_length_counts':{str(k):int(v) for k,v in vals.str.len().value_counts().items()}, 'numeric_fraction':float(pd.to_numeric(vals,errors='coerce').notna().mean())}
for left,right,method in [('T1SERIESINSTANCEUID','SERIESINSTANCEUID','uid'),('LONIIMAGET1','LONIIMAGE','loni'),('SCANDT','STUDYDATE','date')]:
    a=m[['NACCID',left]].copy().rename(columns={left:'key'})
    b=q[['NACCID',right]].copy().rename(columns={right:'key'})
    if method=='date':
        a['key']=pd.to_datetime(a.key,errors='coerce',format='mixed'); b['key']=pd.to_datetime(b.key,errors='coerce',format='mixed')
    else:
        a['key']=a.key.str.strip(); b['key']=b.key.str.strip()
    valid_b=b.dropna().drop_duplicates()
    linked=a.merge(valid_b,on=['NACCID','key'],how='left',indicator=True)
    result[method+'_matched_rows']=int(linked['_merge'].eq('both').sum())
result['shared_participants']=len(set(m.NACCID)&set(q.NACCID))
result['qc_series_types']={str(k):int(v) for k,v in q.SERIESTYPE.value_counts().items()}
result['analysis_project_counts']={str(k):int(v) for k,v in m.PROJECT.value_counts(dropna=False).items()}
result['qc_project_counts']={str(k):int(v) for k,v in q.PROJECT.value_counts(dropna=False).items()}
mq=m.copy(); qq=q.loc[q.SERIESTYPE.eq('T1w') & q.SERIESCHOSEN.eq('1')].copy()
mq['date']=pd.to_datetime(mq.SCANDT,errors='coerce',format='mixed')
qq['date']=pd.to_datetime(qq.STUDYDATE,errors='coerce',format='mixed')
keys=['NACCID','NACCADC','date']
mu=mq.loc[~mq.duplicated(keys,keep=False)]
qu=qq.loc[~qq.duplicated(keys,keep=False)]
pairs=mu.merge(qu,on=keys,how='inner',suffixes=('_mri','_qc'),validate='one_to_one')
result['unique_participant_center_date_selected_t1_pairs']=len(pairs)
result['analysis_rows_on_ambiguous_days']=int(mq.duplicated(keys,keep=False).sum())
result['qc_chosen_t1_rows_on_ambiguous_days']=int(qq.duplicated(keys,keep=False).sum())
result['unique_date_pairs_same_uid']=int(pairs.T1SERIESINSTANCEUID.eq(pairs.SERIESINSTANCEUID).sum())
result['unique_date_pairs_same_loni_image']=int(pairs.LONIIMAGET1.eq(pairs.LONIIMAGE).sum())
result['unique_date_pairs_same_description']=int(pairs.DESCRIPTIONT1.eq(pairs.SERIESDESCRIPTION).sum())
result['unique_date_pairs_by_project']={str(k):int(v) for k,v in pairs.PROJECT_mri.value_counts(dropna=False).items()}
(OUT/'qc_linkage_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
