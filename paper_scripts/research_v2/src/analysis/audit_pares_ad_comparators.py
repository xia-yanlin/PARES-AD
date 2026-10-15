"""User-authorized, protocol-fixed post hoc comparator and transport audit.

No MRI training and no writes to historical result directories. Participant
tables remain under outputs/private. Run prepare, then evaluate, then verify.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research_v2.src.analysis.strengthen_nacc_manuscript import sha, save, interval
from research_v2.src.analysis.evaluate_nacc_external import classification, rr_from_counts

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'research_v2/outputs/private/pares_ad_comparator_transport_2026-10-08'
NACC = ROOT/'research_v2/outputs/private/nacc_external_validation_2026-10-07'
PROTOCOL = ROOT/'research_v2/protocols/pares_ad_comparator_and_transport_2026-10-08.md'
METHODS = ['confidence_only','confidence_plus_conflict','confidence_plus_control','confidence_plus_probability']
BUDGETS = [.05,.10,.20]
PAIRS = [(1,0),(1,2),(1,3),(2,0),(3,0),(3,2)]
COHORTS = ['adni','uid','unique_date_description']
KEY = ['_RID','outer_domain','family','configuration_id']


def design(frame, method):
    p = frame.model_probability_AD.to_numpy(float)
    z = np.abs(np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4))))
    # Historical logit implementation is verified during preparation.
    x = np.column_stack([z,z*z,p>=.5])
    if method == METHODS[0]: return x
    q = frame.pathology_probability_AD.to_numpy(float)
    if method == METHODS[1]: extra = (p>=.5)!=(q>=.5)
    elif method == METHODS[2]: extra = (p>=.5)!=(frame.control_probability_AD.to_numpy()>=.5)
    elif method == METHODS[3]: extra = np.where(p>=.5,1-q,q)
    else: raise ValueError(method)
    return np.column_stack([x,extra])


def fit_scores(source):
    w = len(source)/(source.configuration_id.nunique()*source.groupby('configuration_id').configuration_id.transform('size'))
    y = (source.model_probability_AD.ge(.5)!=source.AD_label).astype(int)
    fits = {}
    for name in METHODS:
        fits[name] = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,solver='lbfgs'))
        fits[name].fit(design(source,name),y,logisticregression__sample_weight=w)
    return fits


def fit_receipt(fits):
    return {name:dict(mean=fit[0].mean_.tolist(),scale=fit[0].scale_.tolist(),
                     coefficients=fit[1].coef_[0].tolist(),intercept=float(fit[1].intercept_[0]),
                     iterations=int(fit[1].n_iter_[0])) for name,fit in fits.items()}


def annotate(frame):
    f = frame.copy()
    f['model_predicted_class'] = f.model_probability_AD.ge(.5).astype(int)
    f['pathology_conflict'] = f.model_predicted_class.ne(f.pathology_probability_AD.ge(.5))
    f['control_conflict'] = f.model_predicted_class.ne(f.control_probability_AD.ge(.5))
    if 'AD_label' in f: f['model_error'] = f.model_predicted_class.ne(f.AD_label).astype(int)
    return f


def prepare():
    import torch
    from research_v2.src.analysis.confirm_pares_ad_independent import (
        BASELINE_CONCEPTS, ASSIGNMENTS, MAPPING, REGISTRY, MANIFEST,
        build_confirmation_cohort, build_pathology_reference, run_black_boxes)
    from research_v2.src.analysis.confirm_pares_ad_state_confirmation import load_development_joined
    from research_v2.src.analysis.validate_pares_ad_conditional_specificity import build_control_reference
    from research_v2.src.analysis.evaluate_pares_ad_error_risk_increment import design_matrix
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'freeze.json').exists(): raise RuntimeError('Prepared freeze exists; use evaluate/verify, do not overwrite it')
    pf = json.loads((NACC/'prediction_freeze/uid/freeze.json').read_text())
    for p,h in pf['source_sha256'].items(): assert sha(ROOT/p)==h, p
    print('Reconstructing unchanged source cross-fitted references.',flush=True)
    dev,_,_,_ = load_development_joined()
    baseline = pd.read_csv(BASELINE_CONCEPTS,dtype={'RID':str},low_memory=False)
    domain = pd.read_csv(ASSIGNMENTS,dtype={'RID':str})
    source_rows = baseline.merge(domain[['RID','acquisition_domain']],on='RID',validate='one_to_one').rename(columns={'RID':'_RID','acquisition_domain':'outer_domain'})
    source_rows = source_rows.loc[source_rows._RID.isin(dev._RID)].copy()
    dev = dev.merge(build_control_reference(source_rows)[['_RID','outer_domain','control_probability_AD']],on=['_RID','outer_domain'],validate='many_to_one')
    assert len(dev)==5184 and dev._RID.nunique()==650
    for name in METHODS[:2]: assert np.allclose(design(dev,name),design_matrix(dev,name).to_numpy(),atol=1e-10)
    print('Replaying source checkpoints with the existing NACC input mask; weights remain fixed.',flush=True)
    missing=json.loads((NACC/'receipt/receipt_summary.json').read_text())['mapping']['absent_fields']
    source_rows[missing]=np.nan
    replay=run_black_boxes(source_rows,pd.read_csv(MANIFEST,dtype={'seed':int}),json.loads(MAPPING.read_text())['nodes'],json.loads(REGISTRY.read_text()))
    masked=dev.drop(columns=['model_probability_AD']).merge(replay[KEY+['model_probability_AD']],on=KEY,validate='one_to_one')
    masked=annotate(masked)
    fits={'full':fit_scores(dev),'masked':fit_scores(masked)}
    receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'protocol_sha256':sha(PROTOCOL),'prepare_script_sha256':sha(Path(__file__)),
             'status':'posthoc_exploratory','prior_outcomes_known':True,
             'nacc_outcomes_opened_in_prepare':False,'source_fits':{k:fit_receipt(v) for k,v in fits.items()},
             'historical_source_hashes':pf['source_sha256'],'files':{},'reproduction':{}}
    columns=KEY+['AD_label','model_probability_AD','pathology_probability_AD','control_probability_AD']
    for name,frame in [('full',dev),('masked',masked)]:
        path=OUT/f'source_{name}_private.csv';frame[columns].to_csv(path,index=False,float_format='%.17g')
        receipt['files'][path.name]=sha(path)
    for name in COHORTS[1:]:
        original=NACC/'prediction_freeze'/name/'predictions.csv'
        old=json.loads((original.parent/'freeze.json').read_text())
        assert sha(original)==old['predictions_sha256']
        p=pd.read_csv(original,dtype={'participant_key':str,'row_key':str,'center':str})
        for method,fit in fits['masked'].items():
            values=fit.predict_proba(design(p,method))[:,1]
            if method in METHODS[:2]:
                diff=float(np.max(np.abs(values-p[method].to_numpy())))
                receipt['reproduction'][name+'_'+method]=diff
                assert diff<1e-6,(name,method,diff)
            p[method]=values
        p['review_weight']=p.weight
        path=OUT/f'{name}_scores_private.csv';p.to_csv(path,index=False,float_format='%.17g')
        receipt['files'][path.name]=sha(path)
        receipt['files'][str(original.relative_to(ROOT))]=sha(original)
    print('Reusing saved ADNI confirmation MRI predictions; reconstructing unchanged source references.',flush=True)
    conf=build_confirmation_cohort(json.loads(MAPPING.read_text())['nodes'])
    cached=ROOT/'research_v2/outputs/private/pares_ad_performance_matched_reference_2026-10-05/confirmation_null_cases.csv'
    saved=pd.read_csv(cached,dtype={'_RID':str})
    # All candidate rows contain the same cached MRI prediction. No candidate is selected.
    assert saved.groupby(KEY).model_probability_AD.nunique().max()==1
    p=saved[KEY+['AD_label','model_probability_AD']].drop_duplicates(KEY)
    assert len(p)==7091 and p._RID.nunique()==882
    assert not set(p._RID)&set(baseline.RID)
    p=p.merge(build_pathology_reference(conf),on=['_RID','outer_domain'],validate='many_to_one')
    p=p.merge(build_control_reference(conf)[['_RID','outer_domain','control_probability_AD']],on=['_RID','outer_domain'],validate='many_to_one')
    p=annotate(p)
    p['participant_key']=p._RID.map(lambda x:hashlib.sha256(('ADNI-audit|'+x).encode()).hexdigest())
    p['row_key']=p.participant_key
    p['source_fold']=p.outer_domain
    p['review_weight']=len(p)/(9*p.groupby('configuration_id').configuration_id.transform('size'))
    p['weight']=1.
    for method,fit in fits['full'].items():p[method]=fit.predict_proba(design(p,method))[:,1]
    path=OUT/'adni_scores_private.csv';p.to_csv(path,index=False,float_format='%.17g')
    receipt['files'][path.name]=sha(path)
    receipt['files'][str(cached.relative_to(ROOT))]=sha(cached)
    save(OUT/'freeze.json',receipt)
    print('Preparation complete. New source score fits and prediction files are frozen.',flush=True)


def load_cohort(name):
    freeze=json.loads((OUT/'freeze.json').read_text())
    assert freeze['protocol_sha256']==sha(PROTOCOL)
    path=OUT/f'{name}_scores_private.csv';assert sha(path)==freeze['files'][path.name]
    frame=pd.read_csv(path,dtype={'participant_key':str,'row_key':str,'center':str,'_RID':str})
    if name!='adni':
        path=NACC/'frozen_cohorts'/name/'outcomes.csv'
        old=json.loads((path.parent/'freeze.json').read_text())
        assert sha(path)==old['outcomes_sha256']
        y=pd.read_csv(path,dtype={'participant_key':str,'row_key':str})
        frame=frame.merge(y,on=['row_key','participant_key'],validate='many_to_one',how='left',indicator=True)
        assert frame['_merge'].eq('both').all()
    return annotate(frame).reset_index(drop=True)


def review_groups(frame,name):
    keys=['configuration_id'] if name=='adni' else ['configuration_id','source_fold']
    groups=[]
    for _,g in frame.groupby(keys,sort=True):
        assert g.participant_key.nunique()==len(g)
        assert g.review_weight.nunique()==1
        orders=[g.sort_values([m,'row_key'],ascending=[False,True],kind='stable').index.to_numpy() for m in METHODS]
        groups.append((float(g.review_weight.iloc[0]),orders))
    return groups


def review_values(frame,name,counts,codes):
    """Counts: bootstrap draws x clustering units. Never split a participant."""
    e=frame.model_error.to_numpy()
    fp=((frame.AD_label==0)&(frame.model_predicted_class==1)).to_numpy()
    fn=((frame.AD_label==1)&(frame.model_predicted_class==0)).to_numpy()
    out=np.zeros((len(counts),len(BUDGETS),len(METHODS),4))
    for weight,orders in review_groups(frame,name):
        for mi,order in enumerate(orders):
            mult=counts[:,codes[order]]
            cumulative=np.cumsum(mult,axis=1)
            ce=np.cumsum(mult*e[order],axis=1)
            cf=np.cumsum(mult*fp[order],axis=1)
            cn=np.cumsum(mult*fn[order],axis=1)
            for bi,budget in enumerate(BUDGETS):
                k=np.floor(cumulative[:,-1]*budget).astype(int)
                idx=np.argmax(cumulative>=k[:,None],axis=1)
                rows=np.arange(len(counts));extra=cumulative[rows,idx]-k
                out[:,bi,mi,0]+=weight*k
                for j,(cum,val) in enumerate([(ce,e),(cf,fp),(cn,fn)],1):
                    out[:,bi,mi,j]+=weight*(cum[rows,idx]-extra*val[order[idx]])
    return out


def review_analysis(frame,name,B=2000):
    pcodes,plevels=pd.factorize(frame.participant_key,sort=True)
    point=review_values(frame,name,np.ones((1,len(plevels)),dtype=int),pcodes)[0]
    w=frame.review_weight.to_numpy();e=frame.model_error.to_numpy()
    fp=((frame.AD_label==0)&(frame.model_predicted_class==1)).to_numpy()
    fn=((frame.AD_label==1)&(frame.model_predicted_class==0)).to_numpy()
    totals=np.array([w.sum(),w@e,w@fp,w@fn])
    rows=[]
    for bi,budget in enumerate(BUDGETS):
        for mi,method in enumerate(METHODS):
            mass,err,sfp,sfn=point[bi,mi]
            rows.append(dict(budget=budget,method=method,yield_=err/mass,actual_review_fraction=mass/totals[0],
                error_capture=err/totals[1],FP_capture=sfp/totals[2],FN_capture=sfn/totals[3],
                remaining_error_rate=(totals[1]-err)/(totals[0]-mass)))
    uncertainty={}
    for unit in (['participant_key'] if name=='adni' else ['participant_key','center']):
        codes,levels=pd.factorize(frame[unit],sort=True);n=len(levels)
        rng=np.random.default_rng(20261018 if unit=='participant_key' else 20261019)
        pieces=[]
        for start in range(0,B,40):
            counts=rng.multinomial(n,np.full(n,1/n),size=min(40,B-start))
            v=review_values(frame,name,counts,codes)
            with np.errstate(divide='ignore',invalid='ignore'):pieces.append(v[...,1]/v[...,0])
            if (start+40)%400==0:print(f'{name} review {unit} {min(start+40,B)}/{B}',flush=True)
        values=np.concatenate(pieces)
        uncertainty[unit]={}
        for a,b in PAIRS:
            label=METHODS[a]+' minus '+METHODS[b]
            uncertainty[unit][label]={str(budget):dict(estimate=float(point[bi,a,1]/point[bi,a,0]-point[bi,b,1]/point[bi,b,0]),
                                                      **interval(values[:,bi,a]-values[:,bi,b])) for bi,budget in enumerate(BUDGETS)}
    configurations={}
    for config,g in frame.groupby('configuration_id',sort=True):
        g=g.reset_index(drop=True);codes,levels=pd.factorize(g.participant_key,sort=True)
        v=review_values(g,name,np.ones((1,len(levels)),dtype=int),codes)[0]
        configurations[config]={str(b):{m:float(v[i,j,1]/v[i,j,0]) for j,m in enumerate(METHODS)} for i,b in enumerate(BUDGETS)}
    return dict(point=rows,intervals=uncertainty,by_configuration=configurations,bootstrap_replicates=B,
                uncertainty_scope='paired evaluation sampling, fixed source fits; nominal exploratory intervals')


def cells(frame,reference,weight):
    y=frame.AD_label.to_numpy(int);p=frame.model_predicted_class.to_numpy(int)
    q=frame[reference].ge(.5).to_numpy(int)
    table=np.bincount(y*4+p*2+q,weights=np.asarray(weight),minlength=8).reshape(2,2,2)
    return table


def cell_metrics(joint):
    joint=np.asarray(joint,float);joint=joint/joint.sum()
    y,p,q=np.indices((2,2,2));flag=p!=q;error=p!=y
    flagged=joint[flag].sum();unflagged=1-flagged
    a=joint[flag&error].sum()/flagged if flagged>0 else np.nan
    b=joint[(~flag)&error].sum()/unflagged if unflagged>0 else np.nan
    total=joint[error].sum()
    result=dict(rr=a/b if b>0 else np.nan,risk_difference=a-b,flagged_fraction=flagged,
                flagged_error=a,unflagged_error=b,error_capture=joint[flag&error].sum()/total if total>0 else np.nan,
                classifier_error=total,reference_error=joint[q!=y].sum(),
                both_correct=joint[(p==y)&(q==y)].sum(),both_wrong=joint[(p!=y)&(q!=y)].sum())
    for pp,qq in [(1,0),(0,1)]:
        sl=(p==pp)&(q==qq);mass=joint[sl].sum()
        result[f'model_{pp}_reference_{qq}_fraction']=mass
        result[f'model_{pp}_reference_{qq}_reference_win']=joint[sl&(q==y)].sum()/mass if mass else np.nan
    return result


def combine(conditional,prevalence):
    return np.asarray(conditional)*np.array([1-prevalence,prevalence])[:,None,None]


def symmetric_decomposition(a,n,pa,pn,metric):
    aa=cell_metrics(combine(a,pa))[metric];an=cell_metrics(combine(a,pn))[metric]
    na=cell_metrics(combine(n,pa))[metric];nn=cell_metrics(combine(n,pn))[metric]
    mix=((an-aa)+(nn-na))/2;conditional=((na-aa)+(nn-an))/2
    assert not np.isfinite(nn-aa) or np.isclose(mix+conditional,nn-aa,atol=1e-10)
    return dict(total=nn-aa,composition=mix,within_diagnosis_joint_behavior=conditional,
                adni_at_adni_prevalence=aa,adni_at_nacc_prevalence=an,
                nacc_at_adni_prevalence=na,nacc_at_nacc_prevalence=nn)


def harmonized_weights(frame):
    configs=frame.groupby('participant_key').configuration_id.transform('nunique')
    folds=frame.groupby(['participant_key','configuration_id']).participant_key.transform('size')
    weight=1/(configs*folds)
    assert np.allclose(weight.groupby(frame.participant_key).sum(),1.)
    return weight.to_numpy()


def transport_analysis(frames):
    out={};conditional={};jointrows=[];configrows=[]
    for name,f in frames.items():
        w=harmonized_weights(f);prevalence=float(f.drop_duplicates('participant_key').AD_label.mean())
        assert np.isclose(np.average(f.AD_label,weights=w),prevalence)
        availability=f.groupby('participant_key').configuration_id.nunique().value_counts().sort_index().to_dict()
        out[name]=dict(participants=int(f.participant_key.nunique()),prevalence=prevalence,
                       configuration_availability=availability,references={})
        conditional[name]={}
        for label,reference in [('AD','pathology_probability_AD'),('control','control_probability_AD')]:
            table=cells(f,reference,w);cond=table/table.sum(axis=(1,2))[:,None,None]
            conditional[name][label]=cond
            old=cell_metrics(cells(f,reference,f.weight))
            out[name]['references'][label]=dict(historical_weight_metrics=old,
                native=cell_metrics(table),at_adni_prevalence=cell_metrics(combine(cond,342/882)))
            for y in [0,1]:
                for p in [0,1]:
                    for q in [0,1]:
                        mask=(f.AD_label==y)&(f.model_predicted_class==p)&(f[reference].ge(.5)==q)
                        jointrows.append(dict(cohort=name,reference=label,diagnosis=y,model_class=p,reference_class=q,
                            participants_with_cell=int(f.loc[mask,'participant_key'].nunique()),model_case_rows=int(mask.sum()),
                            participant_weight=float(table[y,p,q]),conditional_fraction=float(cond[y,p,q])))
            for config,g in f.groupby('configuration_id',sort=True):
                gw=harmonized_weights(g);t=cells(g,reference,gw);c=t/t.sum(axis=(1,2))[:,None,None]
                for mode,j in [('native',t),('standardized',combine(c,342/882))]:
                    configrows.append(dict(cohort=name,reference=label,configuration_id=config,weighting=mode,**cell_metrics(j)))
    decompositions=[]
    for name in COHORTS[1:]:
        for label in ['AD','control']:
            for metric in ['rr','risk_difference']:
                result=symmetric_decomposition(conditional['adni'][label],conditional[name][label],
                    out['adni']['prevalence'],out[name]['prevalence'],metric)
                decompositions.append(dict(comparator=name,reference=label,metric=metric,**result))
    pd.DataFrame(jointrows).to_csv(OUT/'joint_cells_aggregate.csv',index=False)
    pd.DataFrame(configrows).to_csv(OUT/'transport_by_configuration_aggregate.csv',index=False)
    pd.DataFrame(decompositions).to_csv(OUT/'composition_decomposition_aggregate.csv',index=False)
    return dict(cohorts=out,decompositions=decompositions,
                warning='Descriptive, not causal; no cross-cohort independence assumed; actual deployment differs')


def diagnostic_metrics(frame):
    w=harmonized_weights(frame);y=frame.AD_label.to_numpy()
    result={}
    for name,col in [('MRI_classifier','model_probability_AD'),('AD_reference','pathology_probability_AD'),('control','control_probability_AD')]:
        p=frame[col].to_numpy();pred=p>=.5
        result[name]={**classification(y,p,w),'sensitivity':np.average(pred[y==1],weights=w[y==1]),
                      'specificity':np.average(~pred[y==0],weights=w[y==0]),'predicted_ad_fraction':np.average(pred,weights=w)}
    return result


def evaluate():
    assert (OUT/'freeze.json').exists()
    if (OUT/'evaluation.json').exists():raise RuntimeError('Completed evaluation exists; do not overwrite it')
    receipt=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(PROTOCOL),
                 script_sha256=sha(Path(__file__)),freeze_sha256=sha(OUT/'freeze.json'),status='posthoc_exploratory')
    save(OUT/'evaluation_started.json',receipt)
    frames={name:load_cohort(name) for name in COHORTS}
    result={**receipt,'cohorts':{}}
    reviews=[];contrasts=[];diagnostics=[]
    for name,f in frames.items():
        destination=OUT/(name+'_review.json')
        if destination.exists():
            saved=json.loads(destination.read_text())
            assert saved['freeze_sha256']==receipt['freeze_sha256']
            review=saved['review']
        else:
            review=review_analysis(f,name)
            save(destination,dict(freeze_sha256=receipt['freeze_sha256'],review=review))
        metrics={m:classification(f.model_error,f[m],f.review_weight) for m in METHODS}
        diag=diagnostic_metrics(f)
        result['cohorts'][name]=dict(participants=int(f.participant_key.nunique()),predictions=len(f),
                                   review_file=destination.name,error_risk_metrics=metrics,diagnostic_metrics=diag)
        for row in review['point']:reviews.append(dict(cohort=name,**row))
        for unit,comparisons in review['intervals'].items():
            for contrast,budgets in comparisons.items():
                for budget,value in budgets.items():contrasts.append(dict(cohort=name,cluster=unit,contrast=contrast,budget=float(budget),
                    estimate=value['estimate'],ci_low=value['ci95'][0],ci_high=value['ci95'][1],valid_draws=value['valid_draws']))
        for method,values in diag.items():diagnostics.append(dict(cohort=name,method=method,**values))
    pd.DataFrame(reviews).to_csv(OUT/'review_metrics_aggregate.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'review_contrasts_aggregate.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics_aggregate.csv',index=False)
    result['transport']=transport_analysis(frames)
    save(OUT/'evaluation.json',result)
    print('Evaluation complete; independent checks and interpretation remain.',flush=True)


def self_test():
    # Deliberately tied scores, unequal clustering sizes, and partial copies.
    f=pd.DataFrame(dict(participant_key=['a','b','c','d'],row_key=['a','b','c','d'],
                       configuration_id=['x']*4,source_fold=['s']*4,review_weight=[.5]*4,
                       model_predicted_class=[1,0,1,0],AD_label=[0,0,1,1],model_error=[1,0,0,1]))
    for i,m in enumerate(METHODS):f[m]=[.7,.7,.2,.1] if i%2==0 else [.1,.7,.7,.2]
    codes=np.arange(4);counts=np.array([[2,0,3,1],[0,4,0,0],[1,1,1,1]])
    got=review_values(f,'uid',counts,codes)
    for d,count in enumerate(counts):
        for mi,m in enumerate(METHODS):
            order=f.sort_values([m,'row_key'],ascending=[False,True]).index.to_numpy()
            expanded=np.repeat(order,count[order])
            for bi,budget in enumerate(BUDGETS):
                chosen=f.iloc[expanded[:int(len(expanded)*budget)]]
                expected=.5*np.array([len(chosen),chosen.model_error.sum(),((chosen.AD_label==0)&(chosen.model_predicted_class==1)).sum(),((chosen.AD_label==1)&(chosen.model_predicted_class==0)).sum()])
                assert np.allclose(got[d,bi,mi],expected)
    a=np.array([[[.7,.1],[.1,.1]],[[.1,.2],[.1,.6]]])
    n=np.array([[[.8,.05],[.1,.05]],[[.3,.1],[.1,.5]]])
    for metric in ['rr','risk_difference']:
        r=symmetric_decomposition(a,n,.4,.13,metric)
        assert np.isclose(r['total'],r['composition']+r['within_diagnosis_joint_behavior'])
        r=symmetric_decomposition(a,a,.4,.13,metric)
        assert np.isclose(r['within_diagnosis_joint_behavior'],0)
        r=symmetric_decomposition(a,n,.4,.4,metric)
        assert np.isclose(r['composition'],0)
    f['model_probability_AD']=[.8,.2,.6,.3];f['pathology_probability_AD']=[.1,.1,.9,.8];f['control_probability_AD']=[.6,.2,.2,.7]
    changed=f.copy();changed['AD_label']=1-changed.AD_label
    for m in METHODS:assert np.array_equal(design(f,m),design(changed,m))
    assert np.allclose(design(f,METHODS[3])[:,-1],[.9,.1,.1,.8])
    print('Synthetic checks passed: selection multiplicities, ties, zero budgets, label isolation and decomposition.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['test','prepare','evaluate'],required=True)
    args=p.parse_args()
    if args.stage=='test':self_test()
    elif args.stage=='prepare':prepare()
    else:evaluate()


if __name__=='__main__':main()
