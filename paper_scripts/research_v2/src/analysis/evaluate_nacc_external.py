"""Stage C: verify frozen predictions, then evaluate all external endpoints."""
from pathlib import Path
import json
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from research_v2.src.analysis.confirm_pares_ad_independent import sha256_file

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research_v2/outputs/private/nacc_external_validation_2026-10-07'
PROTOCOL=ROOT/'research_v2/protocols/nacc_external_analysis_freeze_2026-10-07.md'
DETAILS=ROOT/'research_v2/protocols/nacc_external_inference_details_2026-10-07.md'
B=2000

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x

def classification(y,p,w):
    p=np.asarray(p,float); y=np.asarray(y,int);w=np.asarray(w,float)
    safe=np.clip(p,1e-6,1-1e-6)
    return {'auroc':roc_auc_score(y,p,sample_weight=w) if len(np.unique(y))==2 else np.nan,
            'auprc':average_precision_score(y,p,sample_weight=w) if len(np.unique(y))==2 else np.nan,
            'brier':np.average((p-y)**2,weights=w),
            'log_loss':np.average(-(y*np.log(safe)+(1-y)*np.log1p(-safe)),weights=w)}

def risk_counts(frame,flag):
    w=frame.weight.to_numpy(); e=frame.model_error.to_numpy();f=frame[flag].to_numpy(bool)
    return np.column_stack([w*f*e,w*f,w*(~f)*e,w*(~f)])

def rr_from_counts(counts):
    with np.errstate(divide='ignore',invalid='ignore'):
        a=counts[...,0]/counts[...,1];b=counts[...,2]/counts[...,3]
        return a/b,a-b,a,b

def interval(values):
    values=np.asarray(values); valid=np.isfinite(values)
    return {'ci95':np.quantile(values[valid],[.025,.975]).tolist() if valid.any() else [None,None],
            'valid_draws':int(valid.sum()),'invalid_draws':int((~valid).sum())}

def primary_bootstrap(frame,unit,seed):
    codes,levels=pd.factorize(frame[unit],sort=True); n=len(levels)
    raw=np.column_stack([risk_counts(frame,'pathology_conflict'),risk_counts(frame,'control_conflict')])
    grouped=np.zeros((n,8));np.add.at(grouped,codes,raw)
    rng=np.random.default_rng(seed)
    multiplicities=rng.multinomial(n,np.full(n,1/n),size=B)
    totals=multiplicities@grouped
    ad=rr_from_counts(totals[:,:4])[0];control=rr_from_counts(totals[:,4:])[0]
    point=rr_from_counts(grouped[:,:4].sum(axis=0))
    cp=rr_from_counts(grouped[:,4:].sum(axis=0))[0]
    return {'resampling_unit':unit,'clusters':n,'ad_rr':float(point[0]),'ad_risk_difference':float(point[1]),
            'conflict_error_rate':float(point[2]),'aligned_error_rate':float(point[3]),'control_rr':float(cp),
            'ad_minus_control_rr':float(point[0]-cp),'ad_rr_interval':interval(ad),
            'ad_minus_control_interval':interval(ad-control)}

class WeightedRank:
    """Pre-sort once; tie-aware weighted AUROC and AP for bootstrap counts."""
    def __init__(self,y,p,w,unit_codes):
        order=np.argsort(-np.asarray(p),kind='stable')
        self.y=np.asarray(y)[order]; self.w=np.asarray(w)[order];self.codes=unit_codes[order]
        scores=np.asarray(p)[order]
        self.ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    def score(self,multiplicities):
        weight=self.w*multiplicities[self.codes]
        tp=np.cumsum(weight*self.y)[self.ends];fp=np.cumsum(weight*(1-self.y))[self.ends]
        if tp[-1]<=0 or fp[-1]<=0:return np.nan,np.nan
        previous_tp=np.r_[0,tp[:-1]];previous_fp=np.r_[0,fp[:-1]]
        auc=np.sum((fp-previous_fp)*(tp+previous_tp)/2)/(tp[-1]*fp[-1])
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        ap=np.sum((tp-previous_tp)*precision)/tp[-1]
        return auc,ap

def secondary_bootstrap(frame):
    codes,units=pd.factorize(frame.participant_key,sort=True);n=len(units)
    y=frame.model_error.to_numpy();w=frame.weight.to_numpy()
    names=['confidence_only','confidence_plus_conflict']
    ranks=[WeightedRank(y,frame[name].to_numpy(),w,codes) for name in names]
    losses=[]
    for name in names:
        p=frame[name].to_numpy();safe=np.clip(p,1e-6,1-1e-6)
        losses.extend([w*(p-y)**2,w*(-y*np.log(safe)-(1-y)*np.log1p(-safe))])
    raw=np.column_stack(losses+[w]);grouped=np.zeros((n,5));np.add.at(grouped,codes,raw)
    rng=np.random.default_rng(20261007);delta=[]
    for b in range(B):
        counts=rng.multinomial(n,np.full(n,1/n))
        a=ranks[0].score(counts);c=ranks[1].score(counts);tot=counts@grouped
        delta.append([c[0]-a[0],c[1]-a[1],(tot[2]-tot[0])/tot[4],(tot[3]-tot[1])/tot[4]])
        if (b+1)%500==0:print(f'Secondary paired bootstrap {b+1}/{B}',flush=True)
    values=np.asarray(delta)
    return {name:interval(values[:,i]) for i,name in enumerate(['auroc','auprc','brier','log_loss'])}

def state_summary(frame):
    aligned=frame.external_state.eq('PATHOLOGY_ALIGNED')
    ref=np.average(frame.loc[aligned,'model_error'],weights=frame.loc[aligned,'weight']) if aligned.any() else np.nan
    rows=[]
    for state,g in frame.groupby('external_state',sort=True):
        rate=np.average(g.model_error,weights=g.weight)
        rows.append({'state':state,'predictions':len(g),'participants':g.participant_key.nunique(),
                     'weighted_fraction':g.weight.sum()/frame.weight.sum(),'error_rate':rate,'rr_vs_aligned':rate/ref if ref>0 else np.nan})
    return rows

def triage(frame):
    totals={}
    for _,group in frame.groupby(['configuration_id','source_fold'],sort=True):
        n=len(group);k=int(group.pathology_conflict.sum())
        methods={'binary_conflict':group.loc[group.pathology_conflict],
                 'confidence_equal_workload':group.assign(distance=(group.model_probability_AD-.5).abs()).sort_values(['distance','row_key'],kind='stable').head(k)}
        for budget in [.05,.10,.20]:
            for name in ['confidence_only','confidence_plus_conflict']:
                methods[f'{name}_{budget:.0%}']=group.sort_values([name,'row_key'],ascending=[False,True],kind='stable').head(int(np.floor(n*budget)))
        for name,selected in methods.items():
            entry=totals.setdefault(name,{'selected_weight':0.,'captured_error_weight':0.,'selected_predictions':0})
            entry['selected_weight']+=selected.weight.sum();entry['captured_error_weight']+=(selected.weight*selected.model_error).sum();entry['selected_predictions']+=len(selected)
    for entry in totals.values():entry['yield']=entry['captured_error_weight']/entry['selected_weight'] if entry['selected_weight'] else np.nan
    return totals

def evaluate(linkage):
    cohort=OUT/'frozen_cohorts'/linkage;predroot=OUT/'prediction_freeze'/linkage
    cf=json.loads((cohort/'freeze.json').read_text());pf=json.loads((predroot/'freeze.json').read_text())
    if cf['analysis_protocol_sha256']!=sha256_file(PROTOCOL) or pf['analysis_protocol_sha256']!=sha256_file(PROTOCOL):raise RuntimeError('Protocol hash changed')
    if sha256_file(predroot/'predictions.csv')!=pf['predictions_sha256']:raise RuntimeError('Prediction hash changed')
    if sha256_file(cohort/'outcomes.csv')!=cf['outcomes_sha256']:raise RuntimeError('Outcome file hash changed')
    destination=OUT/'external_results'/linkage;destination.mkdir(parents=True,exist_ok=True)
    access={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'predictions_sha256':pf['predictions_sha256'],
            'outcomes_sha256':cf['outcomes_sha256'],'analysis_script_sha256':sha256_file(Path(__file__)),
            'inference_details_sha256':sha256_file(DETAILS),'protocol_sha256':sha256_file(PROTOCOL)}
    (destination/'outcome_access.json').write_text(json.dumps(access,indent=2),encoding='utf-8')
    predictions=pd.read_csv(predroot/'predictions.csv',dtype={'row_key':str,'participant_key':str,'center':str})
    outcomes=pd.read_csv(cohort/'outcomes.csv',dtype={'row_key':str,'participant_key':str})
    if outcomes.row_key.duplicated().any():raise RuntimeError('Duplicate outcome keys')
    frame=predictions.merge(outcomes,on=['row_key','participant_key'],validate='many_to_one',how='left',indicator=True)
    if not frame['_merge'].eq('both').all() or len(frame)!=cf['participants']*58:raise RuntimeError('Incomplete outcome join')
    frame['model_error']=frame.model_predicted_class.ne(frame.AD_label).astype(int)
    primary=primary_bootstrap(frame,'participant_key',20261007)
    center=primary_bootstrap(frame,'center',20261008)
    subgroups=[]
    for level in ['configuration_id','family','center','model_predicted_class','AD_label']:
        for value,g in frame.groupby(level,sort=True):
            rr,rd,a,b=rr_from_counts(risk_counts(g,'pathology_conflict').sum(axis=0))
            subgroups.append({'level':level,'value':str(value),'participants':g.participant_key.nunique(),'predictions':len(g),'rr':rr,'risk_difference':rd,'conflict_error_rate':a,'aligned_error_rate':b,
                             'ad_prediction_fraction':np.average(g.model_predicted_class,weights=g.weight),
                             **classification(g.AD_label,g.model_probability_AD,g.weight)})
    config=[r for r in subgroups if r['level']=='configuration_id'];family=[r for r in subgroups if r['level']=='family']
    gates={'rr_at_least_2':primary['ad_rr']>=2,'participant_ci_above_1':primary['ad_rr_interval']['ci95'][0] is not None and primary['ad_rr_interval']['ci95'][0]>1,
           'valid_bootstrap_at_least_95_percent':primary['ad_rr_interval']['valid_draws']>=.95*B,
           'positive_7_of_9_configurations':sum(r['rr']>1 for r in config)>=7,'positive_all_3_families':all(r['rr']>1 for r in family) and len(family)==3}
    unique=frame.drop_duplicates('participant_key')
    classifiers={'black_box':classification(frame.AD_label,frame.model_probability_AD,frame.weight),
                 'ad_reference':classification(unique.AD_label,unique.pathology_probability_AD,np.ones(len(unique))),
                 'control_reference':classification(unique.AD_label,unique.control_probability_AD,np.ones(len(unique)))}
    errors={name:classification(frame.model_error,frame[name],frame.weight) for name in ['confidence_only','confidence_plus_conflict']}
    differences={k:errors['confidence_plus_conflict'][k]-errors['confidence_only'][k] for k in errors['confidence_only']}
    w=frame.weight.to_numpy(); observed=frame.model_logit.to_numpy();fitted=frame.disease_reconstructed_logit.to_numpy()
    behavior={'weighted_coverage':np.average(frame.covered,weights=w),
              'weighted_r2':1-np.sum(w*(observed-fitted)**2)/np.sum(w*(observed-np.average(observed,weights=w))**2),
              'acquisition_attribution':'unavailable_new_program','projection_only':True}
    print(f'Computing all prespecified secondary intervals for {linkage}.',flush=True)
    secondary=secondary_bootstrap(frame)
    summary={'linkage':linkage,'participants':len(unique),'ad':int(unique.AD_label.sum()),'cn':int((unique.AD_label==0).sum()),
             'centers':unique.center.nunique(),'model_cell_predictions':len(frame),'primary':primary,'center_sensitivity':center,
             'replication_gates':gates,'replication_criteria_met':all(gates.values()),'classifiers':classifiers,'behavior':behavior,
             'states':state_summary(frame),'error_risk_models':errors,'error_risk_increment':differences,'error_risk_increment_intervals':secondary,
             'triage':triage(frame),'boundary':'Provisional external-program evaluation; linkage and ADNI overlap limits remain. Disease-only projection, not complete five-state transport.',
             'provenance':access}
    pd.DataFrame(subgroups).to_csv(destination/'subgroup_results.csv',index=False)
    (destination/'summary.json').write_text(json.dumps(clean(summary),indent=2),encoding='utf-8')
    print(json.dumps(clean({'linkage':linkage,'primary':primary,'gates':gates,'secondary_delta':differences}),indent=2),flush=True)

def main():
    started=time.monotonic()
    for linkage in ['uid','unique_date_description']:evaluate(linkage)
    print(f'All frozen external evaluations finished: {time.monotonic()-started:.1f} seconds.',flush=True)

if __name__=='__main__':main()
