"""Independent arithmetic checks for the frozen October 8 exploratory audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from research_v2.src.analysis import audit_pares_ad_comparators as a
from research_v2.src.analysis.strengthen_nacc_manuscript import sha, save, review_bootstrap


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def direct_features(f, method):
    p=f.model_probability_AD.to_numpy();pclip=np.clip(p,.0001,.9999)
    z=np.abs(np.log(pclip)-np.log1p(-pclip))
    parts=[z,z**2,(p>=.5).astype(float)]
    if method==a.METHODS[1]:parts.append((p>=.5)!=(f.pathology_probability_AD.to_numpy()>=.5))
    if method==a.METHODS[2]:parts.append((p>=.5)!=(f.control_probability_AD.to_numpy()>=.5))
    if method==a.METHODS[3]:
        q=f.pathology_probability_AD.to_numpy()
        parts.append(q+(p>=.5)*(1-2*q))
    return np.array(parts,dtype=float).T


def explicit_selection(f,name,counts,codes):
    out=np.zeros((len(counts),3,4,4))
    keys=['configuration_id'] if name=='adni' else ['configuration_id','source_fold']
    for _,g in f.groupby(keys):
        for mi,m in enumerate(a.METHODS):
            order=g.sort_values([m,'row_key'],ascending=[False,True],kind='stable').index.to_numpy()
            for d,mult in enumerate(counts):
                expanded=np.repeat(order,mult[codes[order]])
                for bi,b in enumerate(a.BUDGETS):
                    s=f.loc[expanded[:int(len(expanded)*b)]]
                    w=s.review_weight.to_numpy();y=s.AD_label.to_numpy();p=s.model_predicted_class.to_numpy()
                    out[d,bi,mi]=out[d,bi,mi]+[w.sum(),w@(p!=y),w@((p==1)&(y==0)),w@((p==0)&(y==1))]
    return out


def scalar_metrics(t):
    total=t.sum();flag=error_flag=unflag=error_unflag=0.
    for y in (0,1):
        for p in (0,1):
            for q in (0,1):
                v=t[y,p,q]
                if p!=q:
                    flag+=v
                    if p!=y:error_flag+=v
                else:
                    unflag+=v
                    if p!=y:error_unflag+=v
    rf=error_flag/flag;ru=error_unflag/unflag
    return dict(rr=rf/ru,risk_difference=rf-ru,flagged_fraction=flag/total,
                flagged_error=rf,unflagged_error=ru)


def main():
    freeze=read(a.OUT/'freeze.json');evaluation=read(a.OUT/'evaluation.json')
    assert sha(a.PROTOCOL)==freeze['protocol_sha256']
    assert sha(Path(a.__file__))==freeze['prepare_script_sha256']
    checked={}
    for k,h in {**freeze['historical_source_hashes'],**freeze['files']}.items():
        p=(a.OUT/k) if (a.OUT/k).exists() else a.ROOT/k
        assert sha(p)==h,k
        checked[str(p.relative_to(a.ROOT))]=h
    old_adni=read(a.ROOT/'research_v2/confirmation_results_2026-07-23/summary.json')
    for k,h in old_adni['source_sha256'].items():
        assert sha(Path(k))==h,k
        checked[str(Path(k).relative_to(a.ROOT))]=h
    old_bio=read(a.ROOT/'research_v2/pares_ad_biospecificity_nonmolecular_2026-07-23/summary.json')
    frames={n:a.load_cohort(n) for n in a.COHORTS}
    source=pd.read_csv(a.OUT/'source_full_private.csv',dtype={'_RID':str})
    assert not set(source._RID)&set(frames['adni']._RID)
    gradients={}
    for mode,fits in freeze['source_fits'].items():
        train=pd.read_csv(a.OUT/f'source_{mode}_private.csv')
        weights=len(train)/(train.configuration_id.nunique()*train.groupby('configuration_id').configuration_id.transform('size').to_numpy())
        y=(train.model_probability_AD.ge(.5)!=train.AD_label).to_numpy()
        for m,r in fits.items():
            x=direct_features(train,m)
            assert np.allclose(x.mean(axis=0),r['mean'],atol=1e-12)
            assert np.allclose(x.std(axis=0),r['scale'],atol=1e-12)
            z=(x-r['mean'])/r['scale'];beta=np.asarray(r['coefficients'])
            residual=weights*(expit(z@beta+r['intercept'])-y)
            # Independently differentiate the source weighted logistic objective.
            grad=np.r_[z.T@residual+beta/.1,residual.sum()]/weights.sum()
            gradients[mode+'_'+m]=float(np.max(np.abs(grad)))
            assert gradients[mode+'_'+m]<.0002
    max_score_diff=0.;max_selection_diff=0.;tables={};old_checks={}
    for n,f in frames.items():
        fits=freeze['source_fits']['full' if n=='adni' else 'masked']
        for m,r in fits.items():
            z=(direct_features(f,m)-r['mean'])/r['scale']
            values=expit(z@r['coefficients']+r['intercept'])
            diff=float(np.max(np.abs(values-f[m].to_numpy())))
            max_score_diff=max(max_score_diff,diff);assert diff<1e-12
        point=explicit_selection(f,n,np.ones((1,f.participant_key.nunique()),int),pd.factorize(f.participant_key,sort=True)[0])[0]
        new=read(a.OUT/f'{n}_review.json')
        # Check real weighted selections, duplicate copies and cluster sampling.
        for unit in (['participant_key'] if n=='adni' else ['participant_key','center']):
            codes,levels=pd.factorize(f[unit],sort=True)
            counts=np.random.default_rng(731).multinomial(len(levels),np.full(len(levels),1/len(levels)),size=3)
            direct=explicit_selection(f,n,counts,codes)
            fast=a.review_values(f,n,counts,codes)
            diff=float(np.max(np.abs(direct-fast)));max_selection_diff=max(max_selection_diff,diff)
            assert np.allclose(direct,fast,rtol=1e-12,atol=1e-10)
        review=new['review'] if 'review' in new else new
        for bi,b in enumerate(a.BUDGETS):
            for mi,m in enumerate(a.METHODS):
                row=next(x for x in review['point'] if x['budget']==b and x['method']==m)
                assert np.isclose(point[bi,mi,1]/point[bi,mi,0],row['yield_'],atol=1e-12)
        if n!='adni':
            old=read(a.NACC/'external_results'/n/'summary.json')
            for bi,b in enumerate(a.BUDGETS):
                for mi,m in enumerate(a.METHODS[:2]):
                    assert np.isclose(point[bi,mi,1]/point[bi,mi,0],old['triage'][f'{m}_{b:.0%}']['yield'],atol=1e-12)
        # Independent count-table construction: distribute one vote per person.
        size=f.groupby(['participant_key','configuration_id']).participant_key.transform('size')
        configs=f.groupby('participant_key').configuration_id.transform('nunique')
        w=1./size/configs
        assert np.allclose(w.groupby(f.participant_key).sum(),1.)
        tables[n]={}
        for label,col in [('AD','pathology_probability_AD'),('control','control_probability_AD')]:
            t=np.zeros((2,2,2));old_t=t.copy()
            for y in (0,1):
                for p in (0,1):
                    for q in (0,1):
                        mask=(f.AD_label==y)&(f.model_predicted_class==p)&(f[col].ge(.5)==q)
                        t[y,p,q]=w[mask].sum();old_t[y,p,q]=f.loc[mask,'weight'].sum()
            expected_old=(old_adni['corroboration']['error_risk_ratio'] if label=='AD' else old_bio['nonmolecular_results']['anatomical_specificity']['control_reference']['error_risk_ratio']) if n=='adni' else old['primary']['ad_rr' if label=='AD' else 'control_rr']
            old_rr=scalar_metrics(old_t)['rr'];assert np.isclose(old_rr,expected_old,atol=1e-10)
            old_checks[n+'_'+label]=dict(reproduced=old_rr,historical=expected_old)
            conditional=t/t.sum(axis=(1,2))[:,None,None];tables[n][label]=conditional
            standardized=np.array([conditional[0]*(540/882),conditional[1]*(342/882)])
            result=evaluation['transport']['cohorts'][n]['references'][label]
            for mode,table in [('native',t),('at_adni_prevalence',standardized)]:
                for metric,v in scalar_metrics(table).items():assert np.isclose(v,result[mode][metric],atol=1e-10)
    for row in evaluation['transport']['decompositions']:
        n=row['comparator'];label=row['reference'];metric=row['metric']
        pa=342/882;pn=frames[n].drop_duplicates('participant_key').AD_label.mean()
        def calc(cohort,pr):return scalar_metrics(tables[cohort][label]*np.array([1-pr,pr])[:,None,None])[metric]
        aa,an,na,nn=calc('adni',pa),calc('adni',pn),calc(n,pa),calc(n,pn)
        assert np.isclose((an-aa+nn-na)/2,row['composition'],atol=1e-10)
        assert np.isclose((na-aa+nn-an)/2,row['within_diagnosis_joint_behavior'],atol=1e-10)
        assert np.isclose(nn-aa,row['total'],atol=1e-10)
    # Reproduce the earlier seed's borderline strict-center CI, preserving both.
    legacy=review_bootstrap(frames['uid'],'center')
    historical=read(a.NACC/'manuscript_strengthening/uid/review_summary.json')['intervals']['center']
    for b in legacy:assert np.allclose(legacy[b]['ci95'],historical[b]['ci95'],atol=1e-12)
    for p in a.OUT.glob('*_aggregate.csv'):
        df=pd.read_csv(p)
        assert not set(df.columns)&{'participant_key','row_key','_RID','RID','NACCID'}
        assert not df.astype(str).apply(lambda c:c.str.fullmatch('[a-f0-9]{64}').any()).any()
    r=pd.read_csv(a.OUT/'review_metrics_aggregate.csv');c=pd.read_csv(a.OUT/'review_contrasts_aggregate.csv')
    assert len(r)==36 and len(c)==90 and c.valid_draws.eq(2000).all()
    kit=a.ROOT/'research_v2/human_audit_2026-10-08'
    assignments=pd.read_csv(kit/'assignment_template.csv')
    assert len(assignments)==288 and assignments.status.eq('PLANNED_NOT_RECRUITED').all()
    assert assignments.groupby(['reader_slot','condition']).size().eq(8).all()
    assert assignments.groupby(['case_placeholder','condition']).size().eq(4).all()
    assert not assignments.duplicated(['reader_slot','case_placeholder']).any()
    for p in ['responses_template.csv','adjudication_template.csv','case_manifest_template.csv']:
        assert len(pd.read_csv(kit/p))==0
    save(a.OUT/'verification.json',dict(status='PASS',verifier_sha256=sha(Path(__file__)),
        freeze_sha256=sha(a.OUT/'freeze.json'),evaluation_sha256=sha(a.OUT/'evaluation.json'),
        source_hashes_checked=checked,score_reconstruction_max_absolute_difference=max_score_diff,
        explicit_real_selection_max_absolute_difference=max_selection_diff,historical_rr_reproduction=old_checks,
        source_objective_max_gradient=gradients,
        real_bootstrap_copy_checks='3 draws per cohort/cluster type, all budgets and all methods',
        legacy_uid_center_intervals=legacy,
        seed_boundary_warning='Earlier and current strict-center 10% intervals fall on opposite sides very near zero; no stable significance claim.',
        aggregate_privacy_checks=True,human_kit_balance_and_empty_response_checks=True,
        result_hashes={p.name:sha(p) for p in a.OUT.glob('*_aggregate.csv')}))
    print('PASS: immutable inputs, independent score/rank/cell arithmetic, historical yields/RRs/CIs, aggregate privacy, human kit.')


if __name__=='__main__':main()
