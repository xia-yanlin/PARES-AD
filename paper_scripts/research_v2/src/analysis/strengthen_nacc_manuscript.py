"""Focused post-hoc clinical corroboration and fixed-workload uncertainty.

Frozen MRI predictions are only read. Participant-level outputs remain private.
Run from repository root with the bundled Python (NumPy/Pandas only).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import hashlib
import json
import zipfile
import datetime
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research_v2/outputs/private/nacc_external_validation_2026-10-07'
RAW = ROOT / 'research_v2/data/received/nacc/request_19347_nacc75_202609'
OUT = BASE / 'manuscript_strengthening'
PROTOCOL = ROOT / 'research_v2/protocols/nacc_manuscript_strengthening_2026-10-07.md'
COHORTS = ['uid', 'unique_date_description']
BUDGETS = [.05, .10, .20]
METHODS = ['confidence_only', 'confidence_plus_conflict']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x


def save(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def interval(x):
    x = np.asarray(x); valid = np.isfinite(x)
    return dict(ci95=np.quantile(x[valid], [.025, .975]).tolist() if valid.any() else [None, None],
                valid_draws=int(valid.sum()), invalid_draws=int((~valid).sum()))


def load_frozen(name):
    pp = BASE / 'prediction_freeze' / name / 'predictions.csv'
    op = BASE / 'frozen_cohorts' / name / 'outcomes.csv'
    fp = op.parent / 'features.csv'
    pf = json.loads((pp.parent / 'freeze.json').read_text())
    cf = json.loads((op.parent / 'freeze.json').read_text())
    assert sha(pp) == pf['predictions_sha256']
    assert sha(op) == cf['outcomes_sha256']
    assert sha(fp) == cf['features_sha256']
    p = pd.read_csv(pp, dtype={'participant_key': str, 'row_key': str, 'center': str})
    o = pd.read_csv(op, dtype={'participant_key': str, 'row_key': str})
    f = p.merge(o, on=['participant_key', 'row_key'], validate='many_to_one', how='left', indicator=True)
    assert f['_merge'].eq('both').all() and len(f) == cf['participants'] * 58
    assert f.groupby('participant_key').size().eq(58).all()
    assert np.allclose(f.groupby('participant_key').weight.sum(), 9)
    f['model_error'] = f.model_predicted_class.ne(f.AD_label).astype(int)
    return f, dict(predictions=sha(pp), outcomes=sha(op), features=sha(fp))


def selected_copies(multiplicity_in_rank_order, k):
    """Select first k copies, including a partial tie/copy group at cutoff."""
    before = np.cumsum(multiplicity_in_rank_order, axis=-1) - multiplicity_in_rank_order
    return np.minimum(np.maximum(np.asarray(k)[..., None] - before, 0), multiplicity_in_rank_order)


def review_point(frame):
    results = {}
    e = frame.model_error.to_numpy(); w = frame.weight.to_numpy()
    fp = (frame.model_predicted_class.eq(1) & frame.AD_label.eq(0)).to_numpy()
    fn = (frame.model_predicted_class.eq(0) & frame.AD_label.eq(1)).to_numpy()
    total_error = (w * e).sum(); total_fp = w[fp].sum(); total_fn = w[fn].sum()
    for b in BUDGETS:
        entry = {}
        for method in METHODS:
            mass = err = sfp = sfn = 0.
            for _, g in frame.groupby(['configuration_id', 'source_fold'], sort=True):
                s = g.sort_values([method, 'row_key'], ascending=[False, True], kind='stable').head(int(len(g)*b))
                mass += s.weight.sum(); err += (s.weight*s.model_error).sum()
                sfp += s.loc[s.model_predicted_class.eq(1) & s.AD_label.eq(0), 'weight'].sum()
                sfn += s.loc[s.model_predicted_class.eq(0) & s.AD_label.eq(1), 'weight'].sum()
            entry[method] = dict(yield_=err/mass, actual_review_fraction=mass/w.sum(),
                                all_error_capture=err/total_error, FP_capture=sfp/total_fp if total_fp else np.nan,
                                FN_capture=sfn/total_fn if total_fn else np.nan,
                                remaining_error_rate=(total_error-err)/(w.sum()-mass))
        entry['yield_difference'] = entry[METHODS[1]]['yield_'] - entry[METHODS[0]]['yield_']
        results[str(b)] = entry
    return results


def review_bootstrap(frame, unit, B=2000):
    codes, levels = pd.factorize(frame[unit], sort=True)
    n = len(levels); rng = np.random.default_rng(20261009 if unit == 'participant_key' else 20261010)
    groups = []
    for _, g in frame.groupby(['configuration_id', 'source_fold'], sort=True):
        assert g.participant_key.nunique() == len(g)
        assert g.weight.nunique() == 1
        orders = [g.sort_values([m, 'row_key'], ascending=[False, True], kind='stable').index.to_numpy() for m in METHODS]
        groups.append((float(g.weight.iloc[0]), orders))
    errors = frame.model_error.to_numpy()
    draws = []
    for start in range(0, B, 50):
        counts = rng.multinomial(n, np.full(n, 1/n), size=min(50, B-start))
        numerator = np.zeros((len(counts), len(BUDGETS), 2)); denom = np.zeros((len(counts), len(BUDGETS)))
        for weight, orders in groups:
            for mi, order in enumerate(orders):
                mult = counts[:, codes[order]]
                size = mult.sum(axis=1)
                for bi, budget in enumerate(BUDGETS):
                    k = np.floor(size*budget).astype(int)
                    selected = selected_copies(mult, k)
                    numerator[:, bi, mi] += weight * (selected @ errors[order])
                    if mi == 0: denom[:, bi] += weight*k
        with np.errstate(divide='ignore', invalid='ignore'):
            draws.append((numerator[:, :, 1]-numerator[:, :, 0])/denom)
        if (start+50) % 500 == 0:
            print(f'Review {unit}: {start+50}/{B}', flush=True)
    values = np.concatenate(draws)
    return {str(b): interval(values[:, i]) for i, b in enumerate(BUDGETS)}


def clinical_link(frame):
    """Recover the frozen scan and original diagnostic visit without searching scores."""
    bundle = RAW / '01_current_research_priority/investigator_scan_clariti_mri_csv_nacc75.zip'
    with zipfile.ZipFile(bundle) as z:
        with z.open('investigator_scan_clariti_mrisbm_nacc75.csv') as f:
            mri = pd.read_csv(f, usecols=['NACCID','T1SERIESINSTANCEUID','SCANDT'], dtype=str)
    mri['participant_key'] = mri.NACCID.map(lambda s: hashlib.sha256(('NACC75-PARES-19347|'+s).encode()).hexdigest())
    mri['row_key'] = [hashlib.sha256((p+'|'+u).encode()).hexdigest() for p,u in zip(mri.participant_key, mri.T1SERIESINSTANCEUID)]
    mri = mri.drop_duplicates(['row_key','participant_key','SCANDT'])
    assert not mri.row_key.duplicated().any()
    keep = ['row_key','participant_key','center','AD_label','pathology_logit','control_probability_AD']
    unique = frame[keep].drop_duplicates()
    assert not unique.participant_key.duplicated().any()
    unique = unique.merge(mri[['row_key','participant_key','SCANDT']], on=['row_key','participant_key'], validate='one_to_one', how='left')
    assert unique.SCANDT.notna().all()
    weighted = frame.assign(wz=frame.weight*frame.model_logit).groupby('participant_key').agg(wz=('wz','sum'),w=('weight','sum'))
    unique['mean_model_logit'] = unique.participant_key.map(weighted.wz/weighted.w)
    unique['control_logit'] = np.log(np.clip(unique.control_probability_AD,1e-6,1-1e-6)/(1-np.clip(unique.control_probability_AD,1e-6,1-1e-6)))
    cols = ['NACCID','VISITDATE','NACCUDSD','NACCALZD','NACCAGE','NACCSEX','EDUC','FORMVER','CDRSUM','NACCMMSE']
    uds_path = RAW / '01_current_research_priority/investigator_nacc75.csv'
    uds = pd.read_csv(uds_path, usecols=cols, dtype={'NACCID':str}, low_memory=False)
    uds['participant_key'] = uds.NACCID.map(lambda s: hashlib.sha256(('NACC75-PARES-19347|'+s).encode()).hexdigest())
    uds = uds.loc[uds.participant_key.isin(unique.participant_key)].copy()
    uds['visit_date'] = pd.to_datetime(uds.VISITDATE, format='mixed', errors='coerce')
    groups = dict(tuple(uds.groupby('participant_key', sort=False)))
    records=[]; ambiguous=0
    for row in unique.to_dict('records'):
        g = groups[row['participant_key']]
        scan = pd.to_datetime(row['SCANDT'], format='mixed')
        gap = (g.visit_date-scan).dt.days
        nearest = g.loc[gap.abs().eq(gap.abs().min())]
        assert gap.abs().min() <= 180
        assert len(nearest[['NACCUDSD','NACCALZD']].drop_duplicates()) == 1
        nearest = nearest.loc[nearest.visit_date.eq(nearest.visit_date.min())]
        if len(nearest[cols[2:]].drop_duplicates()) != 1:
            ambiguous += 1; continue
        v = nearest.iloc[0]
        label = 0 if v.NACCUDSD == 1 else 1 if v.NACCUDSD == 4 and v.NACCALZD == 1 else None
        assert label == row['AD_label']
        records.append({**row, **{c:v[c] for c in cols[4:]}, 'gap_days':int((v.visit_date-scan).days)})
    linked = pd.DataFrame(records)
    valid = {'NACCAGE':linked.NACCAGE.between(18,120), 'NACCSEX':linked.NACCSEX.isin([1,2]),
             'EDUC':linked.EDUC.between(0,36),
             'CDRSUM':linked.CDRSUM.between(0,18) & ((linked.CDRSUM*2)%1).eq(0) & ~linked.CDRSUM.isin([16.5,17.5]),
             'NACCMMSE':linked.NACCMMSE.between(0,30) & (linked.NACCMMSE%1).eq(0)}
    for col, mask in valid.items(): linked.loc[~mask, col] = np.nan
    linked['FORMVER'] = linked.FORMVER.astype(str)
    info = dict(frozen_participants=len(unique), linked=len(linked), ambiguous_visit_exclusions=ambiguous,
                diagnostic_agreement=True, median_abs_gap_days=float(linked.gap_days.abs().median()),
                missing={c:int(linked[c].isna().sum()) for c in valid},
                uds_sha256=sha(uds_path), mri_bundle_sha256=sha(bundle))
    return linked, info


def crossfit_setup(data, outcome):
    cols = ['NACCAGE','NACCSEX','EDUC','AD_label','gap_days','mean_model_logit','pathology_logit','control_logit']
    d = data.dropna(subset=cols+[outcome]).sort_values('participant_key').reset_index(drop=True).copy()
    y = d[outcome].to_numpy(float)
    if outcome == 'NACCMMSE': y = 30-y
    numeric = d[['NACCAGE','EDUC','gap_days','mean_model_logit','pathology_logit','control_logit']].to_numpy(float)
    cat = pd.get_dummies(d[['NACCSEX','AD_label','center','FORMVER']].astype(str), drop_first=True, dtype=float).to_numpy()
    folds = np.zeros(len(d), dtype=int)
    for label in [0,1]:
        idx = np.flatnonzero(d.AD_label.to_numpy()==label)
        folds[idx] = np.arange(len(idx))%5
    return d, y, numeric, cat, folds


def clinical_oof(y, numeric, cat, folds, count):
    predictions=np.full((len(y),3),np.nan)
    for f in range(5):
        train=(folds!=f)&(count>0); test=folds==f
        w=count[train].astype(float)
        mu=np.average(numeric[train],axis=0,weights=w)
        sd=np.sqrt(np.average((numeric[train]-mu)**2,axis=0,weights=w));sd[sd<1e-12]=1
        z=(numeric-mu)/sd
        base=np.column_stack([np.ones(len(y)),z[:,:4],cat])
        for j,extra in enumerate([None,4,5]):
            x=base if extra is None else np.column_stack([base,z[:,extra]])
            tr=x[train];xtw=tr.T*w
            penalty=np.eye(x.shape[1]);penalty[0,0]=0
            beta=np.linalg.solve(xtw@tr+penalty,xtw@y[train])
            predictions[test,j]=x[test]@beta
    mean=np.average(y,weights=count)
    sst=np.sum(count*(y-mean)**2)
    if sst<=0:return np.full(3,np.nan),predictions
    return 1-np.sum(count[:,None]*(y[:,None]-predictions)**2,axis=0)/sst,predictions


def clinical_analysis(data, outcome, B=1000):
    d,y,num,cat,folds=crossfit_setup(data,outcome)
    desc=dict(n=len(d),ad=int(d.AD_label.sum()),cn=int((d.AD_label==0).sum()),
              centers=int(d.center.nunique()), excluded_complete_case=len(data)-len(d),
              raw_score_mean=float(d[outcome].mean()),raw_score_sd=float(d[outcome].std()),
              severity_floor_fraction=float(np.mean(y==0)))
    if len(d)<25 or len(np.unique(y))<2:
        return {**desc,'status':'not_estimable'},None
    point,pred=clinical_oof(y,num,cat,folds,np.ones(len(d)))
    rng=np.random.default_rng(20261011);draws=[]
    for b in range(B):
        count=rng.multinomial(len(d),np.full(len(d),1/len(d)))
        r,_=clinical_oof(y,num,cat,folds,count)
        draws.append([r[1]-r[0],r[2]-r[0],r[1]-r[2]])
        if (b+1)%250==0:print(f'Clinical {outcome} n={len(d)}: {b+1}/{B}',flush=True)
    delta=np.array([point[1]-point[0],point[2]-point[0],point[1]-point[2]])
    results={k:dict(estimate=float(delta[i]),**interval(np.asarray(draws)[:,i])) for i,k in enumerate(['AD_increment','control_increment','AD_minus_control'])}
    oof=d[['participant_key','row_key','AD_label']].copy()
    oof['outcome_severity']=y;oof['fold']=folds
    for i,name in enumerate(['base','AD','control']):oof['prediction_'+name]=pred[:,i]
    return {**desc,'status':'exploratory_clinical_corroboration','r2':dict(zip(['base','AD','control'],point)),
            'contrasts':results,'bootstrap':'participant paired; clinical regressions and scaling refitted; folds fixed'},oof


def self_test():
    m=np.array([[2,0,3,1],[0,2,1,0]])
    k=np.array([4,2])
    s=selected_copies(m,k)
    assert np.array_equal(s,np.array([[2,0,2,0],[0,2,0,0]]))
    for mult in [np.array([1,1,1]),np.array([2,0,3]),np.array([0,4,1])]:
        for cutoff in range(mult.sum()+1):
            expanded=np.repeat(np.arange(len(mult)),mult)[:cutoff]
            assert np.array_equal(selected_copies(mult,cutoff),np.bincount(expanded,minlength=len(mult)))
    rng=np.random.default_rng(1);num=rng.normal(size=(80,6));y=rng.normal(size=80)
    cat=(np.arange(80)%2)[:,None];fold=np.arange(80)%5
    p=clinical_oof(y,num,cat,fold,np.ones(80))[1]
    changed=y.copy();changed[fold==0]+=100
    p2=clinical_oof(changed,num,cat,fold,np.ones(80))[1]
    assert np.allclose(p[fold==0],p2[fold==0]), 'held-out outcome entered its own training'
    print('Self-checks passed: bootstrap multiplicity selection and cross-fit outcome isolation.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['link','review','clinical','all','test'],default='all')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    self_test()
    if args.stage=='test':return
    provenance=dict(protocol_sha256=sha(PROTOCOL),script_sha256=sha(Path(__file__)),
                    timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='posthoc_exploratory')
    for name in COHORTS:
        destination=OUT/name;destination.mkdir(exist_ok=True)
        frame,hashes=load_frozen(name)
        if args.stage in ['link','clinical','all']:
            linked,info=clinical_link(frame)
            linked.to_csv(destination/'clinical_linked_private.csv',index=False)
            save(destination/'clinical_linkage.json',{**provenance,**info,'source_hashes':hashes})
            print(f'{name} clinical linkage: '+json.dumps(info),flush=True)
        if args.stage in ['review','all']:
            point=review_point(frame)
            original=json.loads((BASE/'external_results'/name/'summary.json').read_text())
            for budget in BUDGETS:
                for method in METHODS:
                    assert np.isclose(point[str(budget)][method]['yield_'],original['triage'][f'{method}_{budget:.0%}']['yield'],atol=1e-12)
            units={u:review_bootstrap(frame,u) for u in ['participant_key','center']}
            configs={key:review_point(g) for key,g in frame.groupby('configuration_id')}
            save(destination/'review_summary.json',{**provenance,'source_hashes':hashes,'point':point,'intervals':units,'by_configuration':configs})
        if args.stage in ['clinical','all']:
            results={}
            for outcome in ['CDRSUM','NACCMMSE']:
                results[outcome],oof=clinical_analysis(linked,outcome)
                if oof is not None:oof.to_csv(destination/f'{outcome}_oof_private.csv',index=False)
                print(name+' '+outcome+': '+json.dumps(clean(results[outcome])),flush=True)
            save(destination/'clinical_summary.json',{**provenance,'source_hashes':hashes,'clinical_linkage_sha256':sha(destination/'clinical_linked_private.csv'),'outcomes':results})


if __name__=='__main__':main()
