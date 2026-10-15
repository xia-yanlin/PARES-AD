"""Fixed-budget review evaluation with participant- or center-cluster bootstrap.

A review budget b selects floor(b * n) highest-scoring predictions inside each review group (a configuration in
ADNI, a configuration x source-fold unit in NACC). Yield is the share of classifier errors among selected
predictions, with group weights. Bootstrap resamples participants (or centers) with replacement, jointly keeping
all predictions of a resampled unit, and re-ranks the fixed scores in every resample.
"""
import numpy as np
import pandas as pd

BUDGETS = [.05, .10, .20]


def fixed_budget_yield(scores, errors, budget, floor=True):
    """Single-group yield; position breaks ties, zero selected mass returns NaN.

    floor=False retains fractional boundary mass, consistent with Cohort.evaluate.
    This helper is not called by the archived paper analyses.
    """
    scores = np.asarray(scores, float)
    errors = np.asarray(errors, float)
    if scores.ndim != 1 or errors.shape != scores.shape or not len(scores):
        raise ValueError('scores and errors must be nonempty aligned vectors')
    if not np.isfinite(scores).all() or not np.isfinite(errors).all() or not np.isin(errors, [0, 1]).all():
        raise ValueError('scores must be finite and errors binary')
    if not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError('budget must be between zero and one')
    mass = float(np.floor(budget * len(scores))) if floor else float(budget * len(scores))
    if mass == 0:
        return float('nan')
    order = np.lexsort((np.arange(len(scores)), -scores))
    whole = int(np.floor(mass))
    selected_errors = errors[order[:whole]].sum()
    if whole < len(scores):
        selected_errors += (mass - whole) * errors[order[whole]]
    return float(selected_errors / mass)


class Cohort:
    """Evaluation cohort. df needs AD_label, model_probability_AD, participant_key, row_key, review_weight and
    optionally center. group_keys defines review groups (a participant appears at most once per group)."""

    def __init__(self, name, df, group_keys):
        self.name = name
        self.df = df.reset_index(drop=True)
        d = self.df
        self.y = d.AD_label.to_numpy(int)
        pred = d.model_probability_AD.ge(.5).astype(int).to_numpy()
        self.e = (pred != self.y).astype(float)
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
        """counts: draws x units multiplicities. Returns selected [mass, errors, FP, FN] with shape
        (draws, budgets, methods, 4) and totals (draws, 4), weighted by group weight."""
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
    """Point estimate and B bootstrap draws of the yield (selected errors / selected mass)."""
    if cluster == 'participant':
        n = len(coh.plev)
        unit_of_p = np.arange(n)
    else:
        n = coh.ncen
        unit_of_p = coh.cen_of_p
    pt, ptot = coh.evaluate(methods, np.ones((1, n), dtype=int), unit_of_p, prev, floor)
    rng = np.random.default_rng(seed)
    pieces = []
    for start in range(0, B, block):
        counts = rng.multinomial(n, np.full(n, 1 / n), size=min(block, B - start))
        v, _ = coh.evaluate(methods, counts, unit_of_p, prev, floor)
        with np.errstate(divide='ignore', invalid='ignore'):
            pieces.append(v[..., 1] / v[..., 0])
    return pt[0], ptot[0], np.concatenate(pieces)


def summarize(coh_name, scenario, methods, pt, ptot, boots, pairs, cluster, B):
    """Long-format rows: one metric row per score and budget; one contrast row (percentage points, 95% percentile
    interval) per pair of scores (first minus second) and budget."""
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
