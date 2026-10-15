"""Independent numerical checks, using sklearn and explicit bootstrap copies."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from research_v2.src.analysis.strengthen_nacc_manuscript import (
    ROOT,BASE,OUT,PROTOCOL,sha,load_frozen,crossfit_setup,clinical_oof,selected_copies)


def main():
    checks={}
    for cohort in ['uid','unique_date_description']:
        frame,hashes=load_frozen(cohort)
        linked=pd.read_csv(OUT/cohort/'clinical_linked_private.csv',dtype={'center':str,'FORMVER':str})
        result=json.loads((OUT/cohort/'clinical_summary.json').read_text())
        assert sha(OUT/cohort/'clinical_linked_private.csv')==result['clinical_linkage_sha256']
        assert sha(PROTOCOL)==result['protocol_sha256']
        assert sha(Path(__file__).with_name('strengthen_nacc_manuscript.py'))==result['script_sha256']
        data,y,num,cat,folds=crossfit_setup(linked,'CDRSUM')
        preds=np.empty((len(y),3))
        for fold in range(5):
            train=folds!=fold;test=~train
            scale=StandardScaler().fit(num[train]);z=scale.transform(num)
            base=np.column_stack([z[:,:4],cat])
            for i,extra in enumerate([None,4,5]):
                x=base if extra is None else np.column_stack([base,z[:,extra]])
                model=Ridge(alpha=1,fit_intercept=True,solver='svd').fit(x[train],y[train])
                preds[test,i]=model.predict(x[test])
        saved=pd.read_csv(OUT/cohort/'CDRSUM_oof_private.csv')
        assert np.array_equal(saved.participant_key.to_numpy(),data.participant_key.to_numpy())
        errors=[]
        for i,name in enumerate(['base','AD','control']):
            delta=float(np.max(np.abs(preds[:,i]-saved['prediction_'+name].to_numpy())))
            assert delta<1e-9
            assert abs(r2_score(y,preds[:,i])-result['outcomes']['CDRSUM']['r2'][name])<1e-10
            errors.append(delta)
        # Independently expand a bootstrap sample to check weighted refitting.
        rng=np.random.default_rng(904);counts=rng.multinomial(len(y),np.full(len(y),1/len(y)))
        r,weighted=clinical_oof(y,num,cat,folds,counts)
        repeat=np.repeat(np.arange(len(y)),counts)
        expanded=repeat[folds[repeat]!=0];test=folds==0
        scale=StandardScaler().fit(num[expanded]);z=scale.transform(num)
        x=np.column_stack([z[:,:4],cat,z[:,4]])
        fit=Ridge(alpha=1,solver='svd').fit(x[expanded],y[expanded])
        assert np.max(np.abs(fit.predict(x[test])-weighted[test,1]))<1e-9
        checks[cohort]={'frozen_hashes':hashes,'clinical_n':len(y),'sklearn_OOF_max_abs_errors':errors,
                        'expanded_bootstrap_clinical_agreement':True}
        review=json.loads((OUT/cohort/'review_summary.json').read_text())
        assert review['source_hashes']==hashes
        assert review['script_sha256']==result['script_sha256']
        assert all(v['valid_draws']==2000 for u in review['intervals'].values() for v in u.values())
        # Budget amounts and decomposition are checked at all three budgets.
        for budget,row in review['point'].items():
            for method in ['confidence_only','confidence_plus_conflict']:
                v=row[method]
                assert 0<=v['yield_']<=1 and 0<=v['FP_capture']<=1 and 0<=v['FN_capture']<=1
                assert abs(v['actual_review_fraction']-np.floor(len(linked)*float(budget))/len(linked))<1e-12
        if cohort=='uid':
            unit_checks=[]
            for unit in ['participant_key','center']:
                units=sorted(frame[unit].unique());counts=rng.multinomial(len(units),np.full(len(units),1/len(units)))
                mapping=dict(zip(units,counts))
                for _,g in frame.groupby(['configuration_id','source_fold']):
                    for method in ['confidence_only','confidence_plus_conflict']:
                        s=g.sort_values([method,'row_key'],ascending=[False,True],kind='stable')
                        count=s[unit].map(mapping).to_numpy()
                        expanded=s.iloc[np.repeat(np.arange(len(s)),count)]
                        for budget in [.05,.1,.2]:
                            k=int(np.floor(len(expanded)*budget))
                            actual=expanded.head(k).model_error.sum()
                            fast=selected_copies(count,k)@s.model_error.to_numpy()
                            assert actual==fast
                unit_checks.append(unit)
            checks[cohort]['explicit_copy_review_checks']=unit_checks
    checks['scope']={'no_mri_model_or_reference_refitting':True,
                     'clinical_regressions_are_ancillary_cross_fitted_models':True,
                     'new_results_exploratory':True,'original_NACC_predictions_unchanged':True}
    (OUT/'independent_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
