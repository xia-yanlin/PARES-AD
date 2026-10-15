"""Reproduce historical continuous anchors, correct only mismatched CI scale."""
from pathlib import Path
import hashlib
import json
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from research_v2.src.analysis import run_pares_ad_anchor_validation as old
from research_v2.src.analysis.confirm_pares_ad_independent import MAPPING, build_confirmation_cohort
from research_v2.src.analysis.validate_pares_ad_conditional_specificity import build_control_reference

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research_v2/outputs/private/pares_ad_anchor_scale_correction_2026-10-09'
SOURCE=ROOT/'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08/adni_scores_private.csv'
HIST=ROOT/'research_v2/pares_ad_biospecificity_nonmolecular_2026-07-23/summary.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def reconstruct_raw(frame,outcome):
    y=(outcome-outcome.mean())/outcome.std(ddof=0)
    x=old.covariate_design(frame)
    e=np.empty(len(y)); h={k:np.empty(len(y)) for k in ('pathology_logit','control_logit')}
    for train,test in KFold(n_splits=5,shuffle=True,random_state=old.RANDOM_SEED).split(frame):
        beta=np.linalg.lstsq(x[train],y[train],rcond=None)[0]
        e[test]=y[test]-x[test]@beta
        et=y[train]-x[train]@beta
        for k in h:
            q=frame[k].to_numpy(float); q=(q-q.mean())/q.std(ddof=0)
            coefficient=float(q[train]@et/(q[train]@q[train]))
            h[k][test]=coefficient*q[test]
    return e,h

def correction(frame,outcome,historical):
    ez,h,point=old.crossfit_continuous(frame,outcome,('pathology_logit','control_logit'))
    e,hs=reconstruct_raw(frame,outcome)
    assert np.allclose(ez,(e-e.mean())/e.std(ddof=0),atol=1e-12)
    for k in h: assert np.allclose(h[k],hs[k],atol=1e-12)
    generator=np.random.default_rng(old.RANDOM_SEED+20)
    draws_old=[];draws_new=[];maxdiff=0.
    for b in range(old.BOOTSTRAP_REPLICATES):
        ids=generator.integers(0,len(e),size=len(e)); w=np.bincount(ids,minlength=len(e))
        variants=[]
        for residual in [ez,e]:
            vals=[1-float(w@((residual-h[k])**2))/float(w@(residual**2)) for k in h]
            variants.append([*vals,vals[0]-vals[1]])
        draws_old.append(variants[0]);draws_new.append(variants[1])
        if b<25:
            explicit=[1-np.sum((e[ids]-h[k][ids])**2)/np.sum(e[ids]**2) for k in h]
            explicit.append(explicit[0]-explicit[1])
            maxdiff=max(maxdiff,float(np.max(np.abs(np.array(explicit)-variants[1]))))
    oldci=np.quantile(draws_old,[.025,.975],axis=0).T
    newci=np.quantile(draws_new,[.025,.975],axis=0).T
    points=[point['pathology_logit'],point['control_logit'],point['pathology_logit']-point['control_logit']]
    results={}
    for j,k in enumerate(['ad_reference','control_reference','specificity']):
        assert abs(points[j]-historical[k])<1e-10,(k,points[j],historical[k])
        assert np.allclose(oldci[j],historical[k+'_ci95'],atol=1e-10),(k,oldci[j],historical[k+'_ci95'])
        results[k]=dict(estimate=points[j],historical_ci95=oldci[j].tolist(),corrected_ci95=newci[j].tolist())
    # Identity: point effects use exactly the same per-person losses as new CIs.
    ad=1-np.sum((e-h['pathology_logit'])**2)/np.sum(e**2)
    control=1-np.sum((e-h['control_logit'])**2)/np.sum(e**2)
    assert abs(ad-points[0])<1e-12 and abs(control-points[1])<1e-12 and maxdiff<1e-12
    return dict(participants=len(frame),results=results,residual_mean=float(e.mean()),residual_sd=float(e.std(ddof=0)),
                original_points_and_intervals_reproduced=True,explicit_resampling_max_difference=maxdiff,
                bootstrap_replicates=old.BOOTSTRAP_REPLICATES,bootstrap_seed=old.RANDOM_SEED+20,
                interval_scope='fixed cross-fitted predictions; no clinical regression refitting',
                estimand='1 - reference-augmented residual SSE / covariate-only residual SSE')

def main():
    assert not (OUT/'summary.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    # Do not write participants, anchors, or MRI responses to the correction output.
    v=json.loads((SOURCE.parent/'verification.json').read_text())
    assert sha(SOURCE)==v['source_hashes_checked'][str(SOURCE.relative_to(ROOT))]
    hist=json.loads(HIST.read_text())
    conf=build_confirmation_cohort(json.loads(MAPPING.read_text())['nodes'])
    cases=pd.read_csv(SOURCE,dtype={'_RID':str}).sort_values(['_RID','outer_domain','family','configuration_id','AD_label']).reset_index(drop=True)
    control=build_control_reference(conf)[['_RID','outer_domain','control_logit']]
    cases=cases.merge(control,on=['_RID','outer_domain'],validate='many_to_one')
    refs=old.participant_reference(cases,conf).merge(
        conf[['_RID','outer_domain','PHC_Age_T1','phase','manufacturer','field_strength','AD_label']],
        on=['_RID','outer_domain'],validate='one_to_one')
    result={}
    with tempfile.TemporaryDirectory(prefix='anchor_scale_audit_',dir=OUT) as directory:
        directory=Path(directory)
        assert directory.resolve().parent==OUT.resolve(), 'Validate temporary cleanup target.'
        old.export_blind_metadata(directory,old.DEFAULT_RSCRIPT,include_values=True)
        pacc=pd.read_csv(directory/'pacc_values.csv',low_memory=False)
        tau=pd.read_csv(directory/'tau_values.csv',low_memory=False)
        definitions=[('pacc',pacc,'PACC_VISDATE','mPACCtrailsB',180,np.isfinite(pd.to_numeric(pacc.mPACCtrailsB,errors='coerce')),lambda x:-x,'pacc_corroboration'),
            ('tau',tau,'SCANDATE','META_TEMPORAL_SUVR',365,
             tau.TRACER.fillna('').astype(str).str.strip().str.upper().eq('FTP') & pd.to_numeric(tau.qc_flag,errors='coerce').eq(2) & pd.to_numeric(tau.META_TEMPORAL_SUVR,errors='coerce').gt(0),np.log,'tau_primary')]
        for name,values,date,col,days,mask,transform,hkey in definitions:
            linked=old.nearest_anchor(conf,values,date_column=date,value_column=col,window_days=days,extra_mask=mask)
            frame=refs.merge(linked[['_RID','outer_domain','_anchor_value','absolute_gap_days']],on=['_RID','outer_domain'],validate='one_to_one')
            result[name]=correction(frame,transform(frame._anchor_value.to_numpy(float)),hist[hkey]['incremental_r2'])
            assert len(frame)==hist[hkey]['receipt']['participants']
    summary=dict(status='VERIFIED_SCALE_CORRECTION',analyses=result,signed_adjudication_unchanged=True,amyloid_unchanged=True,
        no_mri_inference=True,no_new_hypothesis=True,participant_values_written=False,
        historical_summary_sha256=sha(HIST),historical_code_sha256=sha(Path(old.__file__)),
        correction_code_sha256=sha(Path(__file__)),source_predictions_sha256=sha(SOURCE),
        protocol_sha256=sha(ROOT/'research_v2/protocols/pares_ad_content_correction_2026-10-09.md'))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
