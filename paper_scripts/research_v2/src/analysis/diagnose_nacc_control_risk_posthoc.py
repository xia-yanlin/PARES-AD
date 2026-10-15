"""Explicitly post hoc aggregate diagnosis; no refitting or new thresholds."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research_v2/outputs/private/nacc_external_validation_2026-10-07'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def div(a,b):return float(a/b) if a is not None and b is not None and b!=0 else None

def diagnose(frame):
    y=frame.AD_label.to_numpy(int);m=frame.model_predicted_class.to_numpy(int)
    w=frame.weight.to_numpy(float);err=m!=y;fp=(m==1)&(y==0);fn=(m==0)&(y==1)
    total=w.sum();errors=w[err].sum()
    results={}
    for name,flag,prob in [('AD','pathology_conflict','pathology_probability_AD'),('control','control_conflict','control_probability_AD')]:
        f=frame[flag].to_numpy(bool);ref=frame[prob].to_numpy()>=.5
        assert np.array_equal(f,ref!=m)
        groups={}
        for key,mask in [('flagged',f),('unflagged',~f)]:
            n=w[mask].sum();e=w[mask&err].sum()
            groups[key]={'weighted_prediction_mass':float(n),'prediction_fraction':div(n,total),
                'error_rate':div(e,n),'true_AD_fraction':div(w[mask&(y==1)].sum(),n),
                'predicted_AD_fraction':div(w[mask&(m==1)].sum(),n),
                'FP_fraction_of_errors':div(w[mask&fp].sum(),e),
                'FN_fraction_of_errors':div(w[mask&fn].sum(),e),
                'all_error_capture':div(e,errors),'FP_capture':div(w[mask&fp].sum(),w[fp].sum()),
                'FN_capture':div(w[mask&fn].sum(),w[fn].sum())}
        a,b=groups['flagged']['error_rate'],groups['unflagged']['error_rate']
        results[name]={'groups':groups,'rr':div(a,b),'risk_difference':a-b,
            'reference_AD_prediction_fraction':div(w[ref].sum(),total),
            'reference_sensitivity':div(w[ref&(y==1)].sum(),w[y==1].sum()),
            'reference_specificity':div(w[(~ref)&(y==0)].sum(),w[y==0].sum()),
            'flag_equal_model_predicted_AD_fraction':div(w[f==(m==1)].sum(),total)}
    joint=[]
    a=frame.pathology_conflict.to_numpy(bool);b=frame.control_conflict.to_numpy(bool)
    for label,mask in [('both',a&b),('AD_only',a&~b),('control_only',~a&b),('neither',~a&~b)]:
        mass=w[mask].sum()
        joint.append({'flags':label,'prediction_fraction':div(mass,total),
            'error_rate':div(w[mask&err].sum(),mass),'all_error_capture':div(w[mask&err].sum(),errors),
            'FP_weight':float(w[mask&fp].sum()),'FN_weight':float(w[mask&fn].sum()),
            'correct_weight':float(w[mask&~err].sum())})
    A=results['AD']['groups'];C=results['control']['groups']
    numerator=div(C['flagged']['error_rate'],A['flagged']['error_rate'])
    denominator=div(A['unflagged']['error_rate'],C['unflagged']['error_rate'])
    product=numerator*denominator if numerator is not None and denominator is not None else None
    if product is not None:assert np.isclose(product,div(results['control']['rr'],results['AD']['rr']))
    results.update({'joint_flags':joint,'RR_control_over_AD_decomposition':{
        'flagged_risk_factor':numerator,'inverse_unflagged_risk_factor':denominator,
        'product':product},'model_error_rate':div(errors,total),
        'model_AD_prediction_fraction':div(w[m==1].sum(),total)})
    return results

def main():
    output={'status':'posthoc_descriptive','changes_to_frozen_analysis':'none','cohorts':{}}
    for name in ['uid','unique_date_description']:
        pred=BASE/'prediction_freeze'/name/'predictions.csv';out=BASE/'frozen_cohorts'/name/'outcomes.csv'
        pf=json.loads((pred.parent/'freeze.json').read_text());cf=json.loads((out.parent/'freeze.json').read_text())
        assert sha(pred)==pf['predictions_sha256'] and sha(out)==cf['outcomes_sha256']
        frame=pd.read_csv(pred).merge(pd.read_csv(out),on=['row_key','participant_key'],validate='many_to_one')
        assert len(frame)==cf['participants']*58
        r=diagnose(frame)
        frozen=json.loads((BASE/'external_results'/name/'summary.json').read_text())
        assert np.isclose(r['AD']['rr'],frozen['primary']['ad_rr'])
        assert np.isclose(r['control']['rr'],frozen['primary']['control_rr'])
        r['by_configuration']={key:diagnose(g) for key,g in frame.groupby('configuration_id')}
        r['by_true_class']={str(key):diagnose(g) for key,g in frame.groupby('AD_label')}
        r['source_hashes']={'predictions':sha(pred),'outcomes':sha(out)}
        output['cohorts'][name]=r
    target=BASE/'posthoc_control_diagnosis';target.mkdir(exist_ok=True)
    (target/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ['AD','control','RR_control_over_AD_decomposition','joint_flags']} for k,v in output['cohorts'].items()},indent=2))

if __name__=='__main__':main()
