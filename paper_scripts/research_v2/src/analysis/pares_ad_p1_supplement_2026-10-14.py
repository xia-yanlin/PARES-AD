"""PARES-AD P1 supplement (user-authorized 2026-10-14; post hoc / exploratory).

Prevalence reweighting of the A-minus-C review gain (P1-a) and flag-rate-matched
controls / candidate references (P1-b). Only numpy + pandas are used (no sklearn).
No MRI training or inference. Aggregate outputs only; participant rows are never
written. See research_v2/protocols/pares_ad_p1_prevalence_and_matched_control_2026-10-14.md
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BUDGETS = [.05, .10, .20]
CAND = ['624', '773', '846', '151', '3', '662', '550', '461', '56', '730']
STAGES = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- logistic head
def fit_head(X, y, w, C=0.1, iters=200):
    """sklearn-equivalent: StandardScaler (ddof=0, unweighted) + L2 logistic,
    objective 0.5*||b||^2 + C*sum w_i logloss_i, unpenalized intercept (Newton)."""
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    n, d = Z.shape
    A = np.column_stack([np.ones(n), Z])
    beta = np.zeros(d + 1)
    reg = np.eye(d + 1)
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(A @ beta)))
        g = C * (A.T @ (w * (p - y))) + reg @ beta
        H = C * (A.T @ (A * (w * p * (1 - p))[:, None])) + reg
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return dict(mu=mu, sd=sd, beta=beta)


def predict_head(fit, X):
    Z = (X - fit['mu']) / fit['sd']
    return 1.0 / (1.0 + np.exp(-(fit['beta'][0] + Z @ fit['beta'][1:])))


def design(df, extra=None):
    p = df.model_probability_AD.to_numpy(float)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.abs(np.log(pc / (1 - pc)))
    cols = [z, z * z, (p >= .5).astype(float)]
    if extra is not None:
        cols.append(np.asarray(extra, float))
    return np.column_stack(cols)


def src_weights(src):
    return len(src) / (src.configuration_id.nunique() * src.groupby('configuration_id').configuration_id.transform('size')).to_numpy()


def src_target(src):
    return (src.model_probability_AD.ge(.5) != src.AD_label).astype(float).to_numpy()


def extras(df, kind, thr=None, cand=None):
    p = df.model_probability_AD.to_numpy(float)
    q = df.pathology_probability_AD.to_numpy(float)
    if kind == 'C':
        return None
    if kind == 'A':
        return (p >= .5) != (q >= .5)
    if kind == 'K':
        return (p >= .5) != (df.control_probability_AD.to_numpy(float) >= .5)
    if kind == 'P':
        return np.where(p >= .5, 1 - q, q)
    if kind == 'Kp':
        return (p >= .5) != (df.control_probability_AD.to_numpy(float) >= thr)
    if kind == 'Kc':
        return (p >= .5) != (df[cand].to_numpy(float) >= thr)
    if kind == 'Pk':
        qk = df.control_probability_AD.to_numpy(float)
        return np.where(p >= .5, 1 - qk, qk)
    if kind == 'Pc':
        qc = df[cand].to_numpy(float)
        return np.where(p >= .5, 1 - qc, qc)
    raise ValueError(kind)


def match_threshold(src, w, prob_col):
    p = src.model_probability_AD.to_numpy(float) >= .5
    qa = src.pathology_probability_AD.to_numpy(float) >= .5
    target = float((w * (p != qa)).sum() / w.sum())
    grid = np.round(np.arange(0.01, 0.9901, 0.005), 3)
    prob = src[prob_col].to_numpy(float)
    rates = np.array([(w * (p != (prob >= t))).sum() / w.sum() for t in grid])
    k = int(np.argmin(np.abs(rates - target)))
    return float(grid[k]), float(rates[k]), target


# ----------------------------------------------------------------- review engine
class Cohort:
    def __init__(self, name, df, group_keys):
        self.name = name
        self.df = df.reset_index(drop=True)
        d = self.df
        self.y = d.AD_label.to_numpy(int)
        self.e = (d.model_probability_AD.ge(.5).astype(int).to_numpy() != self.y).astype(float)
        pred = d.model_probability_AD.ge(.5).astype(int).to_numpy()
        self.fp = ((self.y == 0) & (pred == 1)).astype(float)
        self.fn = ((self.y == 1) & (pred == 0)).astype(float)
        self.pcode, self.plev = pd.factorize(d.participant_key, sort=True)
        yl = np.full(len(self.plev), -1)
        yl[self.pcode] = self.y
        assert (yl[self.pcode] == self.y).all(), 'participant label not constant'
        self.ylab = yl
        self.rk = pd.factorize(d.row_key, sort=True)[0]
        self.groups = []
        for _, g in d.groupby(group_keys, sort=True):
            assert g.participant_key.nunique() == len(g), 'participant repeated within a review group'
            assert g.review_weight.nunique() == 1
            self.groups.append((float(g.review_weight.iloc[0]), g.index.to_numpy()))
        if 'center' in d:
            c = pd.factorize(d.center, sort=True)[0]
            cp = np.full(len(self.plev), -1)
            cp[self.pcode] = c
            assert (cp[self.pcode] == c).all(), 'participant spans centers'
            self.cen_of_p = cp
            self.ncen = int(c.max()) + 1
        self.orders = {}

    def set_scores(self, scores):
        for name, s in scores.items():
            s = np.asarray(s, float)
            self.orders[name] = [idx[np.lexsort((self.rk[idx], -s[idx]))] for _, idx in self.groups]

    def evaluate(self, methods, counts, unit_of_p, prev=None, floor=True):
        """counts: draws x units multiplicities. Returns (D,budgets,methods,4) selected
        [mass, errors, FP, FN] and totals (D,4), each weighted by group weight."""
        D = counts.shape[0]
        pc = counts[:, unit_of_p].astype(float)
        if prev is None:
            fAD = np.ones(D)
            fCN = np.ones(D)
        else:
            S = pc.sum(1)
            share = (pc @ (self.ylab == 1).astype(float)) / S
            fAD = prev / share
            fCN = (1 - prev) / (1 - share)
        out = np.zeros((D, len(BUDGETS), len(methods), 4))
        tot = np.zeros((D, 4))
        rows = np.arange(D)
        for gi, (w, idx) in enumerate(self.groups):
            y_i = self.y[idx]
            m_all = pc[:, self.pcode[idx]] * np.where(y_i == 1, fAD[:, None], fCN[:, None])
            tot += w * np.stack([m_all.sum(1), m_all @ self.e[idx], m_all @ self.fp[idx], m_all @ self.fn[idx]], 1)
            for mi, name in enumerate(methods):
                o = self.orders[name][gi]
                m = pc[:, self.pcode[o]] * np.where(self.y[o] == 1, fAD[:, None], fCN[:, None])
                cm = np.cumsum(m, axis=1)
                cum = [np.cumsum(m * v[o], axis=1) for v in (self.e, self.fp, self.fn)]
                for bi, b in enumerate(BUDGETS):
                    target = np.floor(cm[:, -1] * b) if floor else cm[:, -1] * b
                    idx_k = np.argmax(cm >= target[:, None], axis=1)
                    over = cm[rows, idx_k] - target
                    out[:, bi, mi, 0] += w * target
                    for j, (c_arr, v) in enumerate(zip(cum, (self.e, self.fp, self.fn)), 1):
                        out[:, bi, mi, j] += w * (c_arr[rows, idx_k] - over * v[o][idx_k])
        return out, tot


def interval(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def run_boot(coh, methods, cluster, B, seed, prev=None, floor=True, block=40):
    if cluster == 'participant':
        n = len(coh.plev)
        unit_of_p = np.arange(n)
    else:
        n = coh.ncen
        unit_of_p = coh.cen_of_p
    point_counts = np.ones((1, n), dtype=int)
    pt, ptot = coh.evaluate(methods, point_counts, unit_of_p, prev, floor)
    rng = np.random.default_rng(seed)
    pieces = []
    for start in range(0, B, block):
        counts = rng.multinomial(n, np.full(n, 1 / n), size=min(block, B - start))
        v, _ = coh.evaluate(methods, counts, unit_of_p, prev, floor)
        with np.errstate(divide='ignore', invalid='ignore'):
            pieces.append(v[..., 1] / v[..., 0])
    return pt[0], ptot[0], np.concatenate(pieces)


def summarize(coh_name, scenario, methods, pt, ptot, boots, pairs, cluster, B):
    rows = []
    yields = pt[..., 1] / pt[..., 0]
    for bi, b in enumerate(BUDGETS):
        for mi, m in enumerate(methods):
            rows.append(dict(kind='metric', cohort=coh_name, scenario=scenario, cluster=cluster, method=m, budget=b,
                             yield_pct=100 * yields[bi, mi], error_capture_pct=100 * pt[bi, mi, 1] / ptot[1],
                             fp_capture_pct=100 * pt[bi, mi, 2] / ptot[2], fn_capture_pct=100 * pt[bi, mi, 3] / ptot[3],
                             review_fraction_pct=100 * pt[bi, mi, 0] / ptot[0]))
    for a, b_ in pairs:
        ia, ib = methods.index(a), methods.index(b_)
        for bi, b in enumerate(BUDGETS):
            est = 100 * (yields[bi, ia] - yields[bi, ib])
            lo, hi = interval(100 * (boots[:, bi, ia] - boots[:, bi, ib]))
            rows.append(dict(kind='contrast', cohort=coh_name, scenario=scenario, cluster=cluster, contrast=f'{a} minus {b_}',
                             budget=b, estimate_pp=est, ci_low_pp=lo, ci_high_pp=hi, B=B))
    return rows


# ----------------------------------------------------------------- data
def load_all(root):
    priv = root / 'research_v2/outputs/private'
    tr = priv / 'pares_ad_comparator_transport_2026-10-08'
    nacc = priv / 'nacc_external_validation_2026-10-07'
    dt = dict(participant_key=str, row_key=str, center=str, _RID=str)
    data = {}
    adni = pd.read_csv(tr / 'adni_scores_private.csv', dtype=dt)
    data['adni'] = adni
    for name in ['uid', 'unique_date_description']:
        s = pd.read_csv(tr / f'{name}_scores_private.csv', dtype=dt)
        y = pd.read_csv(nacc / 'frozen_cohorts' / name / 'outcomes.csv', dtype=dict(participant_key=str, row_key=str))
        s = s.merge(y, on=['row_key', 'participant_key'], validate='many_to_one', how='left', indicator=True)
        assert s['_merge'].eq('both').all()
        data[name] = s.drop(columns='_merge')
    dtr = dict(_RID=str)
    data['src_full'] = pd.read_csv(tr / 'source_full_private.csv', dtype=dtr)
    data['src_masked'] = pd.read_csv(tr / 'source_masked_private.csv', dtype=dtr)
    # candidate references
    rep = pd.read_csv(priv / 'pares_ad_fixed_candidate_repair_2026-10-09/source_reference_predictions_private.csv', dtype=dict(RID=str))
    rep = rep.rename(columns={'RID': '_RID'})
    cn = pd.read_csv(priv / 'pares_ad_performance_matched_reference_2026-10-05/confirmation_null_cases.csv', dtype=dict(_RID=str, candidate_id=str))
    data['cand_src'] = rep
    data['cand_conf'] = cn
    return data


METHOD_ALIAS = {'C': 'confidence_only', 'A': 'confidence_plus_conflict', 'K': 'confidence_plus_control', 'P': 'confidence_plus_probability'}


def prepare_cohort(df, name):
    d = df.copy()
    d['model_probability_AD'] = d.model_probability_AD.astype(float)
    if name == 'adni':
        keys = ['configuration_id']
    else:
        keys = ['configuration_id', 'source_fold']
    return Cohort(name, d, keys)


def refit_heads(src, kinds):
    w = src_weights(src)
    y = src_target(src)
    heads = {}
    for k, kw in kinds.items():
        kind = kw['kind']
        ex = extras(src, kind, kw.get('thr'), kw.get('cand'))
        heads[k] = fit_head(design(src, ex), y, w)
    return heads


def head_scores(heads, df, kinds):
    return {k: predict_head(h, design(df, extras(df, kinds[k]['kind'], kinds[k].get('thr'), kinds[k].get('cand'))))
            for k, h in heads.items()}


# ----------------------------------------------------------------- stages
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--Bc', type=int, default=1000)
    ap.add_argument('--stages', default='gate,p1a,p1b')
    ap.add_argument('--protocol', default='')
    ap.add_argument('--scen', default='')
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stages = args.stages.split(',')
    data = load_all(root)
    base = ['C', 'A', 'K', 'P']
    base_kinds = {k: dict(kind=k) for k in base}
    cohorts = {}
    for name in ['adni', 'uid', 'unique_date_description']:
        cohorts[name] = prepare_cohort(data[name], name)
    receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='posthoc_exploratory',
                   script_sha256=sha(Path(__file__)), protocol_sha256=sha(Path(args.protocol)) if args.protocol else None,
                   B_participant=args.B, B_center=args.Bc)
    all_rows = []

    # ---------- gate: reproduce saved scores
    heads_full = refit_heads(data['src_full'], base_kinds)
    heads_masked = refit_heads(data['src_masked'], base_kinds)
    gate = {}
    for name in cohorts:
        heads = heads_full if name == 'adni' else heads_masked
        sc = head_scores(heads, data[name], base_kinds)
        gate[name] = {k: float(np.max(np.abs(sc[k] - data[name][METHOD_ALIAS[k]].to_numpy(float)))) for k in base}
    receipt['gate_score_reproduction_max_abs_diff'] = gate
    print('gate score reproduction:', json.dumps(gate), flush=True)
    bad = {n: v for n, d in gate.items() for n, v in d.items() if v > 1e-5}
    if bad:
        print('WARNING: head refit differs from saved scores beyond 1e-5:', bad, flush=True)

    # saved scores are the authoritative fixed scores
    for name, coh in cohorts.items():
        coh.set_scores({k: data[name][METHOD_ALIAS[k]].to_numpy(float) for k in base})

    # native replication of D8 (floor-count rule, original seeds)
    if 'gate' in stages:
        for name, coh in cohorts.items():
            pt, ptot, bo = run_boot(coh, base, 'participant', args.B, 20261018, None, True)
            rows = summarize(name, 'native', base, pt, ptot, bo, [('A', 'C'), ('A', 'K'), ('P', 'C'), ('K', 'C')], 'participant', args.B)
            all_rows += rows
            if name != 'adni':
                pt, ptot, bo = run_boot(coh, base, 'center', args.Bc, 20261019, None, True)
                all_rows += summarize(name, 'native', base, pt, ptot, bo, [('A', 'C'), ('A', 'K'), ('P', 'C'), ('K', 'C')], 'center', args.Bc)
            print(name, 'native done', round(time.time() - t0), 's', flush=True)

    # ---------- P1-a
    if 'p1a' in stages:
        shares = {}
        for name in ['adni', 'uid', 'unique_date_description']:
            c = cohorts[name]
            shares[name] = float((c.ylab == 1).mean())
        receipt['ad_share_participants'] = shares
        print('AD shares', shares, flush=True)
        scen = [('adni', 'rw_to_unique_date_description', shares['unique_date_description']),
                ('adni', 'rw_to_uid', shares['uid']),
                ('adni', 'rw_to_adni_identity', shares['adni']),
                ('unique_date_description', 'rw_to_adni', shares['adni']),
                ('unique_date_description', 'rw_to_uid', shares['uid']),
                ('uid', 'rw_to_adni', shares['adni'])]
        for k, (name, label, prev) in enumerate(scen):
            if args.scen and label + '@' + name not in args.scen.split(','):
                continue
            coh = cohorts[name]
            pairs = [('A', 'C'), ('P', 'C'), ('K', 'C'), ('A', 'K')]
            pt, ptot, bo = run_boot(coh, base, 'participant', args.B, 20261014 + k, prev, False)
            all_rows += summarize(name, label, base, pt, ptot, bo, pairs, 'participant', args.B)
            if name != 'adni':
                pt, ptot, bo = run_boot(coh, base, 'center', args.Bc, 20261114 + k, prev, False)
                all_rows += summarize(name, label, base, pt, ptot, bo, pairs, 'center', args.Bc)
            print('p1a', name, label, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(all_rows).to_csv(out / 'p1a_results.csv', index=False, float_format='%.6g')

    # ---------- P1-b (all heads refit with the same Newton solver for internal consistency)
    if 'p1b' in stages:
        b_rows = []
        thr_info = {}
        n_kinds = {'Cn': dict(kind='C'), 'An': dict(kind='A'), 'Kn': dict(kind='K'), 'Pn': dict(kind='P'), 'Pkn': dict(kind='Pk')}
        for name in ['adni', 'uid', 'unique_date_description']:
            src = data['src_full'] if name == 'adni' else data['src_masked']
            w = src_weights(src)
            t, rate, target = match_threshold(src, w, 'control_probability_AD')
            feasible = abs(rate - target) <= 0.01
            thr_info[name] = dict(min_gap_control_threshold=t, achieved_source_flag_rate=rate, ad_source_flag_rate=target,
                                  matching_feasible=bool(feasible))
            kinds = dict(n_kinds)
            kinds['Kmin'] = dict(kind='Kp', thr=t)
            h = refit_heads(src, kinds)
            sc = head_scores(h, data[name], kinds)
            cohorts[name].set_scores(sc)
            rw = data[name].review_weight.to_numpy(float) if name != 'adni' else np.ones(len(data[name]))
            e = cohorts[name].e
            for fname, ex in [('AD', extras(data[name], 'A')), ('control_0.5', extras(data[name], 'K')),
                              ('control_min_flag', extras(data[name], 'Kp', t))]:
                f = np.asarray(ex, bool)
                rr = (rw[f] @ e[f] / rw[f].sum()) / (rw[~f] @ e[~f] / rw[~f].sum())
                thr_info[name]['rr_' + fname] = float(rr)
                thr_info[name]['flag_rate_' + fname] = float(rw[f].sum() / rw.sum())
            pr = data[name].model_probability_AD.to_numpy(float) >= .5
            yy = cohorts[name].y
            thr_info[name]['classifier_specificity'] = float(((~pr) & (yy == 0)).sum() / (yy == 0).sum())
            thr_info[name]['classifier_sensitivity'] = float((pr & (yy == 1)).sum() / (yy == 1).sum())
        receipt['control_matching'] = thr_info
        print('control matching', json.dumps(thr_info), flush=True)
        m5 = ['Cn', 'An', 'Kn', 'Kmin', 'Pn', 'Pkn']
        pairs = [('An', 'Cn'), ('Kn', 'Cn'), ('Kmin', 'Cn'), ('Pkn', 'Cn'), ('Pn', 'Cn'),
                 ('An', 'Kn'), ('An', 'Kmin'), ('Pn', 'Pkn')]
        for k, name in enumerate(['adni', 'uid', 'unique_date_description']):
            coh = cohorts[name]
            pt, ptot, bo = run_boot(coh, m5, 'participant', args.B, 20261030 + k, None, True)
            b_rows += summarize(name, 'control_comparators', m5, pt, ptot, bo, pairs, 'participant', args.B)
            if name != 'adni':
                pt, ptot, bo = run_boot(coh, m5, 'center', args.Bc, 20261130 + k, None, True)
                b_rows += summarize(name, 'control_comparators', m5, pt, ptot, bo, pairs, 'center', args.Bc)
            print('p1b control', name, round(time.time() - t0), 's', flush=True)
        # candidates (ADNI only)
        src = data['src_full'].merge(data['cand_src'][['_RID', 'outer_domain'] + ['candidate_' + c for c in CAND]],
                                     on=['_RID', 'outer_domain'], how='left', validate='many_to_one')
        assert src['candidate_624'].notna().all(), 'candidate source probabilities missing'
        cn = data['cand_conf']
        adni = data['adni'].copy()
        for c in CAND:
            sub = cn.loc[cn.candidate_id == c, ['_RID', 'outer_domain', 'null_probability']].drop_duplicates()
            assert sub.groupby(['_RID', 'outer_domain']).size().max() == 1
            sub = sub.rename(columns={'null_probability': 'candidate_' + c})
            adni = adni.merge(sub, on=['_RID', 'outer_domain'], how='left', validate='many_to_one')
            assert adni['candidate_' + c].notna().all()
        assert (adni.row_key.to_numpy() == data['adni'].row_key.to_numpy()).all()
        w = src_weights(src)
        scores_c = {}
        cand_info = {}
        for c in CAND:
            col = 'candidate_' + c
            kk = {'Kc' + c: dict(kind='Kc', thr=.5, cand=col), 'Pc' + c: dict(kind='Pc', cand=col)}
            h = refit_heads(src, kk)
            scores_c.update(head_scores(h, adni, kk))
            t, rate, target = match_threshold(src, w, col)
            flag = (adni.model_probability_AD.to_numpy(float) >= .5) != (adni[col].to_numpy(float) >= .5)
            e = cohorts['adni'].e
            cand_info[c] = dict(flag_count_native=int(flag.sum()), flag_rate_native=float(flag.mean()),
                                rr_native=float(e[flag].mean() / e[~flag].mean()),
                                min_gap_threshold=t, achieved_source_flag_rate=rate, ad_source_flag_rate=target,
                                matching_feasible=bool(abs(rate - target) <= 0.01))
        receipt['candidate_info_adni'] = cand_info
        cohorts['adni'].set_scores(scores_c)
        mc = ['Cn', 'An', 'Pn'] + sorted(scores_c.keys())
        pairs = [('An', 'Cn')] + [(k, 'Cn') for k in sorted(scores_c)] + [('An', 'Kc' + c) for c in CAND] + [('Pn', 'Pc' + c) for c in CAND]
        pt, ptot, bo = run_boot(cohorts['adni'], mc, 'participant', args.B, 20261050, None, True)
        b_rows += summarize('adni', 'candidates', mc, pt, ptot, bo, pairs, 'participant', args.B)
        print('p1b candidates', round(time.time() - t0), 's', flush=True)
        pd.DataFrame(b_rows).to_csv(out / 'p1b_results.csv', index=False, float_format='%.6g')

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out / 'native_and_p1a_results.csv', index=False, float_format='%.6g')
    receipt['runtime_seconds'] = round(time.time() - t0)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print('DONE', receipt['runtime_seconds'], 's', flush=True)


if __name__ == '__main__':
    main()
