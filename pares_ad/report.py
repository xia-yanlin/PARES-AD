"""Plain-text case report: classifier and reference probabilities, itemized contributions, residual check.

Rules reproduced from the manuscript: the classifier prediction is always shown; the reference probability and
its itemized contributions are shown beside it; contributions are withheld when the absolute residual of the
output-fitting regression exceeds the empirical 90% bound; the intercept plus the contributions reconstruct the
reference log-odds exactly.
"""
import math
from .reference import CORE_FEATURES

LABELS = {'hippocampal_atrophy': 'Hippocampal atrophy', 'entorhinal_thinning': 'Entorhinal thinning',
          'ventricular_enlargement': 'Ventricular enlargement', 'temporoparietal_thinning': 'Temporoparietal thinning'}


def case_report(case_id, classifier_probability, reference_intercept, severities, coefficients, residual, residual_bound,
                observed_diagnosis=None):
    """Return the report text for one prediction.

    severities, coefficients: sequences ordered as CORE_FEATURES (severity in training SD units, oriented so that
    larger means more AD-like change). residual is classifier logit minus the output-fitting regression value.
    """
    severities, coefficients = list(severities), list(coefficients)
    if len(severities) != len(CORE_FEATURES) or len(coefficients) != len(CORE_FEATURES):
        raise ValueError('exactly four severities and coefficients are required')
    values = [classifier_probability, reference_intercept, residual, residual_bound, *severities, *coefficients]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('report numeric inputs must be finite')
    if not 0 <= classifier_probability <= 1 or residual_bound < 0:
        raise ValueError('invalid probability or residual bound')
    contrib = [s * c for s, c in zip(severities, coefficients)]
    logit = reference_intercept + sum(contrib)
    if not math.isfinite(logit):
        raise ValueError('reference logit overflow')
    if logit >= 0:
        p_ref = 1 / (1 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        p_ref = exp_logit / (1 + exp_logit)
    p_clf = float(classifier_probability)
    pred_clf, pred_ref = ('AD' if p_clf >= .5 else 'CN'), ('AD' if p_ref >= .5 else 'CN')
    covered = abs(residual) <= residual_bound
    lines = [f'Case {case_id}']
    if observed_diagnosis:
        lines.append(f'Observed diagnosis (retrospective): {observed_diagnosis}')
    lines += [f'Classifier AD probability: {p_clf:.1%} -> {pred_clf}',
              f'Anatomical reference AD probability: {p_ref:.1%} -> {pred_ref}',
              'Prediction comparison: ' + ('classes agree' if pred_clf == pred_ref else 'classes DISAGREE (classifier %s, reference %s)' % (pred_clf, pred_ref)),
              f'Residual check: |residual| = {abs(residual):.3f} vs bound {residual_bound:.3f} -> ' + ('contributions shown' if covered else 'contributions WITHHELD (prediction retained)')]
    if covered:
        lines.append('Reference log-odds contributions (positive pushes toward AD):')
        for name, s, c in zip(CORE_FEATURES, severities, contrib):
            lines.append(f'  {LABELS[name]:<26} severity {s:+.2f} SD   contribution {c:+.3f}')
        lines.append(f'  Intercept {reference_intercept:+.3f}; sum = reference log-odds {logit:+.3f}')
    return '\n'.join(lines)
