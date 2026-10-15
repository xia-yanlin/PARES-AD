"""Stage B: source-only fitting and outcome-free NACC prediction freeze.

Never opens NACC outcomes.csv. The new-program disease-only projection is
explicitly different from the original ADNI acquisition-dependent states.
"""
from pathlib import Path
import json
import datetime
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from research_v2.src.analysis.confirm_pares_ad_independent import (
    BASELINE_CONCEPTS, ASSIGNMENTS, MAPPING, REGISTRY, MANIFEST, JOBS_ROOT,
    sha256_file, restore_transformer, run_black_boxes, find_configuration,
    build_neural_model, model_scores, verify_completed_directory,
)
from research_v2.src.analysis.confirm_pares_ad_state_confirmation import load_development_joined
from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS, CONCEPT_ORIENTATION, fit_monotone_pathology_model,
    conformal_absolute_quantile, logit,
)
from research_v2.src.analysis.validate_pares_ad_conditional_specificity import build_control_features
from research_v2.src.analysis.evaluate_pares_ad_error_risk_increment import design_matrix
from research_v2.src.audit.concepts import build_concept_target

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research_v2/outputs/private/nacc_external_validation_2026-10-07'
PROTOCOL=ROOT/'research_v2/protocols/nacc_external_analysis_freeze_2026-10-07.md'
KEY=['_RID','outer_domain','family','configuration_id']
SEVERITIES=[f'{c}_severity' for c in CORE_CONCEPTS]

def build_source_bundle():
    mapping=json.loads(MAPPING.read_text())['nodes']
    registry=json.loads(REGISTRY.read_text())
    manifest=pd.read_csv(MANIFEST,dtype={'seed':int})
    if len(manifest)!=290 or manifest.groupby(['family','configuration_id','outer_domain']).ngroups!=58:
        raise RuntimeError('Source manifest changed')
    development,cohort,assignments,_=load_development_joined()
    baseline=pd.read_csv(BASELINE_CONCEPTS,dtype={'RID':str},low_memory=False)
    domains=pd.read_csv(ASSIGNMENTS,dtype={'RID':str})
    replay=baseline.merge(domains[['RID','acquisition_domain']],on='RID',validate='one_to_one').rename(columns={'RID':'_RID','acquisition_domain':'outer_domain'})
    replay=replay.loc[replay._RID.isin(development._RID)].copy()
    missing=json.loads((OUT/'receipt/receipt_summary.json').read_text())['mapping']['absent_fields']
    replay[missing]=np.nan
    masked=run_black_boxes(replay,manifest,mapping,registry)
    development=development.drop(columns=['model_probability_AD']).merge(masked[KEY+['model_probability_AD']],on=KEY,validate='one_to_one')
    development['model_logit']=logit(development.model_probability_AD)
    development['model_predicted_class']=development.model_probability_AD.ge(.5).astype(int)
    development['pathology_conflict']=development.model_predicted_class.ne(development.pathology_probability_AD.ge(.5).astype(int))
    development['model_error']=development.model_predicted_class.ne(development.AD_label).astype(int)
    behavior={}
    for (family,config),group in development.groupby(['family','configuration_id'],sort=True):
        residuals=[]
        for domain in sorted(group.outer_domain.unique()):
            train=group.loc[group.outer_domain.ne(domain)]
            test=group.loc[group.outer_domain.eq(domain)]
            fit=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(train[SEVERITIES],train.model_logit)
            residuals.extend(np.abs(test.model_logit.to_numpy()-fit.predict(test[SEVERITIES])))
        behavior[(family,config)]={'estimator':make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(group[SEVERITIES],group.model_logit),
                                  'bound':conformal_absolute_quantile(np.asarray(residuals),.9)}
    weights=len(development)/(9*development.groupby('configuration_id').configuration_id.transform('size'))
    risk={}
    for name in ['confidence_only','confidence_plus_conflict']:
        estimator=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,solver='lbfgs'))
        estimator.fit(design_matrix(development,name),development.model_error,logisticregression__sample_weight=weights)
        risk[name]=estimator
    return mapping,registry,manifest,baseline,behavior,risk

def references(baseline,features):
    train_values=[]; eval_values=[]
    for concept in CORE_CONCEPTS:
        target=build_concept_target(concept,baseline,features)
        if not target.train_valid.all() or not target.evaluation_valid.all(): raise RuntimeError('Incomplete core concept')
        train_values.append(CONCEPT_ORIENTATION[concept]*target.train_values)
        eval_values.append(CONCEPT_ORIENTATION[concept]*target.evaluation_values)
    train=np.column_stack(train_values); evaluation=np.column_stack(eval_values)
    intercept,beta=fit_monotone_pathology_model(train,baseline.AD_label.to_numpy(dtype=int),l2_penalty=.1)
    contribution=evaluation*beta
    ref=pd.DataFrame({'row_key':features.row_key,'participant_key':features.participant_key,'center':features.center,
                      'pathology_logit':intercept+contribution.sum(axis=1),'pathology_intercept':intercept})
    ref['pathology_probability_AD']=expit(ref.pathology_logit)
    for j,c in enumerate(CORE_CONCEPTS):
        ref[c+'_severity']=evaluation[:,j]; ref[c+'_contribution']=contribution[:,j]
    control_train,control_eval=build_control_features(baseline,features)
    ci,cb=fit_monotone_pathology_model(control_train,baseline.AD_label.to_numpy(dtype=int),l2_penalty=.1)
    ref['control_probability_AD']=expit(ci+control_eval@cb)
    if not np.allclose(logit(ref.pathology_probability_AD,epsilon=1e-15),ref.pathology_logit,atol=1e-7):
        raise RuntimeError('Reference reconstruction/probability mismatch')
    return ref,{'ad_intercept':intercept,'ad_coefficients':beta.tolist(),'control_intercept':ci,'control_coefficients':cb.tolist()}

def predict_cells(features,mapping,registry,manifest):
    frames=[]; device=torch.device('cpu')
    for (family,config,fold),jobs in manifest.groupby(['family','configuration_id','outer_domain'],sort=True):
        probabilities=[]
        if len(jobs)!=5: raise RuntimeError('Five source seeds required')
        for record in jobs.to_dict('records'):
            directory=JOBS_ROOT/str(record['fw5_job_id'])
            if not verify_completed_directory(directory,expected_job_id=str(record['fw5_job_id'])): raise RuntimeError('Invalid source checkpoint receipt')
            checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
            for field in ['family','configuration_id','outer_domain','seed']:
                if str(checkpoint[field])!=str(record[field]): raise RuntimeError('Checkpoint metadata mismatch')
            transformed=restore_transformer(directory/'preprocessor.npz',mapping).transform(features)
            configuration=find_configuration(registry,{**record,'model_kind':'neural'})
            model=build_neural_model(family,configuration,n_nodes=transformed.shape[1],n_features=transformed.shape[2]).to(device)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            adjacency=np.load(directory/'adjacency.npy') if (directory/'adjacency.npy').exists() else None
            scores=model_scores(model,transformed,adjacency,device)
            platt=json.loads((directory/'summary.json').read_text())['platt']
            probabilities.append(expit(float(platt['coefficient'])*scores+float(platt['intercept'])))
        matrix=np.stack(probabilities)
        frames.append(pd.DataFrame({'row_key':features.row_key,'family':family,'configuration_id':config,
            'source_fold':fold,'model_probability_AD':matrix.mean(axis=0),'seed_probability_sd':matrix.std(axis=0,ddof=1)}))
    return pd.concat(frames,ignore_index=True)

def main():
    started=time.monotonic(); torch.set_num_threads(4)
    print('Fitting source-only deployment rules.',flush=True)
    mapping,registry,manifest,baseline,behavior,risk=build_source_bundle()
    provenance={'analysis_protocol_sha256':sha256_file(PROTOCOL),'script_sha256':sha256_file(Path(__file__)),
        'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in [BASELINE_CONCEPTS,ASSIGNMENTS,MAPPING,REGISTRY,MANIFEST]},
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'nacc_outcomes_opened':False}
    for linkage in ['uid','unique_date_description']:
        source=OUT/'frozen_cohorts'/linkage
        freeze=json.loads((source/'freeze.json').read_text())
        if freeze['analysis_protocol_sha256']!=provenance['analysis_protocol_sha256']: raise RuntimeError('Protocol changed after cohort freeze')
        if sha256_file(source/'features.csv')!=freeze['features_sha256']: raise RuntimeError('Feature freeze mismatch')
        destination=OUT/'prediction_freeze'/linkage
        destination.mkdir(parents=True,exist_ok=True)
        features=pd.read_csv(source/'features.csv',dtype={'row_key':str,'participant_key':str,'center':str},low_memory=False)
        if any(c in features for c in ['AD_label','NACCUDSD','NACCALZD']): raise RuntimeError('Outcomes in prediction features')
        ref,coefficients=references(baseline,features)
        print(f'Predicting {linkage}: {len(features)} participants, 58 source cells.',flush=True)
        predictions=predict_cells(features,mapping,registry,manifest).merge(ref,on='row_key',validate='many_to_one')
        if len(predictions)!=len(features)*58: raise RuntimeError('Prediction cells incomplete')
        predictions['model_logit']=logit(predictions.model_probability_AD)
        predictions['model_predicted_class']=predictions.model_probability_AD.ge(.5).astype(int)
        predictions['pathology_conflict']=predictions.model_predicted_class.ne(predictions.pathology_probability_AD.ge(.5).astype(int))
        predictions['control_conflict']=predictions.model_predicted_class.ne(predictions.control_probability_AD.ge(.5).astype(int))
        for (family,config),index in predictions.groupby(['family','configuration_id']).groups.items():
            bundle=behavior[(family,config)]
            predictions.loc[index,'disease_reconstructed_logit']=bundle['estimator'].predict(predictions.loc[index,SEVERITIES])
            predictions.loc[index,'residual_bound']=bundle['bound']
        predictions['covered']=np.abs(predictions.model_logit-predictions.disease_reconstructed_logit)<=predictions.residual_bound
        predictions['external_state']=np.where(~predictions.covered,'UNEXPLAINED_ABSTAIN',np.where(predictions.pathology_conflict,'PATHOLOGY_CONFLICT','PATHOLOGY_ALIGNED'))
        predictions['acquisition_attribution']='UNAVAILABLE_NEW_PROGRAM'
        fold_counts=manifest.groupby('configuration_id').outer_domain.nunique()
        predictions['weight']=1.0/predictions.configuration_id.map(fold_counts)
        for name,fit in risk.items(): predictions[name]=fit.predict_proba(design_matrix(predictions,name))[:,1]
        numeric=predictions.select_dtypes(include='number')
        if not np.isfinite(numeric.to_numpy()).all(): raise RuntimeError('Non-finite predictions')
        if not np.allclose(predictions.groupby('participant_key').weight.sum(),9): raise RuntimeError('Participant weighting mismatch')
        path=destination/'predictions.csv'
        predictions.to_csv(path,index=False,float_format='%.17g')
        receipt={**provenance,'linkage':linkage,'participants':len(features),'predictions':len(predictions),
                 'features_sha256':freeze['features_sha256'],'predictions_sha256':sha256_file(path),
                 'reference_coefficients':coefficients,'state_contract':'source-trained disease-only deployment projection; acquisition unavailable',
                 'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (destination/'freeze.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(f'Frozen {linkage}; no external outcomes accessed.',flush=True)
    print(f'Prediction stage complete in {time.monotonic()-started:.1f} seconds.',flush=True)

if __name__=='__main__':main()
