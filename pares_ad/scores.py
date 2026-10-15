"""Review scores: confidence (C), binary disagreement (A), control (K), continuous opposition (P), committee (CE).

All scores are StandardScaler + L2 logistic regression (C = 0.1) predicting classifier error, fitted once to source
predictions with equal total weight per configuration and then held fixed. Implemented with numpy only (Newton
solver, unpenalized intercept); results agree with sklearn to solver tolerance.
"""
import numpy as np
import pandas as pd


def fit_head(X, y, w, C=0.1, iters=200):
    """sklearn-equivalent: StandardScaler (ddof=0, unweighted) + L2 logistic, objective
    0.5*||b||^2 + C*sum_i w_i logloss_i with an unpenalized intercept."""
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
    """Inputs of the confidence score C: |z|, |z|^2 and the predicted class, z = logit of the classifier probability;
    an optional extra column gives A, K, P, ..."""
    p = df.model_probability_AD.to_numpy(float)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.abs(np.log(pc / (1 - pc)))
    cols = [z, z * z, (p >= .5).astype(float)]
    if extra is not None:
        cols.append(np.asarray(extra, float))
    return np.column_stack(cols)


def opposition(classifier_probability, reference_probability):
    """Continuous reference opposition O (Eq. 6): reference support for the class opposite to the prediction."""
    p = np.asarray(classifier_probability, float)
    q = np.asarray(reference_probability, float)
    return np.where(p >= .5, 1 - q, q)


def extras(df, kind, ref_col='pathology_probability_AD', control_col='control_probability_AD'):
    p = df.model_probability_AD.to_numpy(float)
    if kind == 'C':
        return None
    if kind == 'A':   # binary disagreement with the AD reference
        return (p >= .5) != (df[ref_col].to_numpy(float) >= .5)
    if kind == 'K':   # binary disagreement with the designated control
        return (p >= .5) != (df[control_col].to_numpy(float) >= .5)
    if kind == 'P':   # continuous opposition from the AD reference
        return opposition(p, df[ref_col].to_numpy(float))
    raise ValueError(kind)


def source_weights(src, config_col='configuration_id'):
    """Equal total weight per configuration (mean weight 1)."""
    n_cfg = src[config_col].nunique()
    return len(src) / (n_cfg * src.groupby(config_col)[config_col].transform('size')).to_numpy()


def source_target(src):
    """Classifier error at threshold 0.5 against the diagnostic label."""
    return (src.model_probability_AD.ge(.5) != src.AD_label).astype(float).to_numpy()


def committee_features(df, key='participant_key'):
    """Participant-level spread of the classifiers' predictions: population SD of the predicted AD probability and
    normalized binary vote disagreement 4 f (1 - f), f = share of predictions of AD, returned per row.
    It is 0 for unanimous and 1 for evenly split predictions. Every available row contributes, including
    configuration x source-fold rows in NACC (58 per participant)."""
    p = df.model_probability_AD.to_numpy(float)
    k = df[key].to_numpy()
    sd = pd.Series(p).groupby(k).transform(lambda s: s.std(ddof=0)).to_numpy(float)
    f = pd.Series((p >= .5).astype(float)).groupby(k).transform('mean').to_numpy(float)
    return sd, 4 * f * (1 - f)


def _rank_by_participant(df, key, v):
    t = pd.DataFrame({'k': df[key].to_numpy(), 'v': v}).drop_duplicates('k').set_index('k')
    return df[key].map(t['v'].rank(pct=True)).to_numpy(float)


def committee_design(df, key='participant_key', use_vote=True, rank=False):
    """Inputs of the committee score CE (C inputs + committee spread); rank=True gives percentile-ranked inputs (CEr)."""
    p = df.model_probability_AD.to_numpy(float)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.abs(np.log(pc / (1 - pc)))
    sd, vote = committee_features(df, key)
    if rank:
        sd, vote = _rank_by_participant(df, key, sd), _rank_by_participant(df, key, vote)
    cols = [z, z * z, (p >= .5).astype(float), sd]
    if use_vote:
        cols.append(vote)
    return np.column_stack(cols)


def crossfit_scores(df, folds=5, seed=20261014, weight_col='review_weight'):
    """Locally refit heads for C, CE, A and P by participant cross-fitting with the cohort's own error labels.

    A label-using comparator for a site that can collect local labels. Participants are assigned to five folds at
    random (unstratified); all rows of a participant stay in one fold. Training rows are weighted by `weight_col`
    (configuration-equal weights, as in the protocol and the reported results); weight_col=None gives equal row
    weights. Every training partition must contain both error classes.
    """
    pk = df.participant_key.to_numpy()
    uniq = pd.unique(pk)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, folds, len(uniq))))
    fold = np.array([fold_of[k] for k in pk])
    w = df[weight_col].to_numpy(float) if weight_col else np.ones(len(df))
    w = w / w.mean()
    y = source_target(df)
    designs = {'Cx': design(df), 'CEx': committee_design(df), 'Ax': design(df, extras(df, 'A')), 'Px': design(df, extras(df, 'P'))}
    out = {k: np.zeros(len(df)) for k in designs}
    for f in range(folds):
        tr, te = fold != f, fold == f
        for k, X in designs.items():
            h = fit_head(X[tr], y[tr], w[tr] / w[tr].mean())
            out[k][te] = predict_head(h, X[te])
    return out
