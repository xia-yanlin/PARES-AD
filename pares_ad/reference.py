"""Four-feature anatomical reference with sign-constrained coefficients and the output-fitting residual gate.

Feature columns follow the manuscript: hippocampal atrophy, entorhinal thinning, ventricular enlargement and
temporoparietal thinning, each oriented so that a larger value means more AD-like change and expressed in
training-population standard deviations. The reference is a logistic model whose four coefficients are
constrained to be nonnegative (L-BFGS-B bounds), so every contribution coefficient * severity pushes toward AD
when the structure is more abnormal.
"""
import math
import numpy as np

CORE_FEATURES = ('hippocampal_atrophy', 'entorhinal_thinning', 'ventricular_enlargement', 'temporoparietal_thinning')


def fit_signed_reference(features, labels, l2_penalty=0.1):
    """Logistic loss with nonnegative coefficients and an unpenalized intercept.

    Objective: mean(logaddexp(0, f) - y f) + 0.5 * l2_penalty * ||beta||^2, f = intercept + X beta, beta >= 0.
    Returns (intercept, coefficients).
    """
    from scipy.optimize import minimize          # scipy is only needed for this function
    from scipy.special import expit
    x = np.asarray(features, float)
    y = np.asarray(labels, float)
    if x.ndim != 2 or x.shape[1] != len(CORE_FEATURES):
        raise ValueError('features must have four columns')
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid inputs')
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError('both classes are required')
    prev = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    init = np.r_[math.log(prev / (1 - prev)), np.full(x.shape[1], 0.1)]

    def objective(par):
        b0, b = par[0], par[1:]
        lin = b0 + x @ b
        loss = float(np.mean(np.logaddexp(0.0, lin) - y * lin)) + 0.5 * l2_penalty * float(b @ b)
        err = expit(lin) - y
        grad = np.r_[float(err.mean()), x.T @ err / len(x) + l2_penalty * b]
        return loss, grad

    res = minimize(lambda p: objective(p)[0], init, jac=lambda p: objective(p)[1], method='L-BFGS-B',
                   bounds=[(None, None)] + [(0.0, None)] * x.shape[1], options={'maxiter': 2000, 'ftol': 1e-12})
    if not res.success:
        raise RuntimeError(res.message)
    coef = np.asarray(res.x[1:], float)
    if (coef < -1e-10).any():
        raise RuntimeError('coefficient bound violated')
    return float(res.x[0]), coef


def reference_logit(features, intercept, coefficients):
    """Return (logit, contributions). logit = intercept + contributions.sum(axis=1) exactly."""
    x = np.asarray(features, float)
    contrib = x * np.asarray(coefficients, float)
    return intercept + contrib.sum(axis=1), contrib


def conformal_absolute_quantile(residuals, coverage=0.90):
    """Empirical residual threshold at rank min(n, ceil((n+1)*coverage)).
    The historical function name is retained for compatibility; this finite-sample rank
    rule alone does not establish split-conformal or cross-domain coverage guarantees."""
    v = np.asarray(residuals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise ValueError('no finite residuals')
    rank = min(len(v), int(math.ceil((len(v) + 1) * coverage)))
    return float(np.partition(v, rank - 1)[rank - 1])


def _ridge(X, y, alpha):
    """sklearn Ridge(alpha) with an unpenalized intercept."""
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, float(ym - xm @ w)


class _Decomposition:
    def __init__(self, mu, sd, w1, b1, w2, b2):
        self.mu, self.sd, self.w1, self.b1, self.w2, self.b2 = mu, sd, w1, b1, w2, b2

    def components(self, structural, acquisition):
        """Return (disease component, acquisition component); their sum plus the residual is the classifier logit."""
        z = (np.asarray(structural, float) - self.mu) / self.sd
        disease = self.b1 + z @ self.w1
        acq = self.b2 + np.asarray(acquisition, float) @ self.w2
        return disease, acq


def fit_output_decomposition(structural, acquisition, classifier_logit, alpha_structural=1.0, alpha_acquisition=1.0):
    """Two-stage fit of the classifier logit: standardized structural terms (ridge), then acquisition one-hot
    terms (ridge) on the residual. Matches StandardScaler + Ridge(alpha=1) followed by Ridge(alpha=1)."""
    S = np.asarray(structural, float)
    A = np.asarray(acquisition, float)
    y = np.asarray(classifier_logit, float)
    mu = S.mean(0)
    sd = S.std(0)
    sd[sd == 0] = 1.0
    w1, b1 = _ridge((S - mu) / sd, y, alpha_structural)
    disease = b1 + ((S - mu) / sd) @ w1
    w2, b2 = _ridge(A, y - disease, alpha_acquisition)
    return _Decomposition(mu, sd, w1, b1, w2, b2)


def reliability_state(covered, acquisition_component, disease_component, pathology_conflict):
    """Display state used by the case report (acquisition threshold 0.25 logit)."""
    acq_dom = bool(abs(acquisition_component) >= 0.25 and abs(acquisition_component) >= abs(disease_component))
    if not covered:
        return 'UNEXPLAINED_ABSTAIN', acq_dom
    if acq_dom:
        return 'ACQUISITION_SENSITIVE', acq_dom
    if pathology_conflict:
        return 'PATHOLOGY_CONFLICT', acq_dom
    return 'PATHOLOGY_ALIGNED', acq_dom
