"""PARES-AD: Prespecified Anatomical Reference for Error Screening in Alzheimer's disease.

Reference fitting, review scores, fixed-budget evaluation and case reports.
Only numpy, pandas and scipy are required.
"""
from .reference import (CORE_FEATURES, fit_signed_reference, reference_logit, conformal_absolute_quantile,
                        fit_output_decomposition, reliability_state)
from .scores import (fit_head, predict_head, design, extras, opposition, committee_features, committee_design,
                     source_weights, source_target, crossfit_scores)
from .review import BUDGETS, Cohort, run_boot, summarize, fixed_budget_yield
from .report import case_report

__version__ = '1.1.0'
