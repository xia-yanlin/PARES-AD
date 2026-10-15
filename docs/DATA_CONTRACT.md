# Data contract

## Scores and fitting

All probability inputs must be finite and between zero and one. Labels are binary AD=1/CN=0.
The reference requires four source-standardized, oriented features, in `CORE_FEATURES` order; preprocessing must be
learned on the permitted source partition. The library does not load raw images or implement the full FreeSurfer pipeline.

For fixed-budget scores, `model_probability_AD` supplies p; clip p to [1e-4, 1-1e-4] when computing z=logit(p).
C uses [abs(z), abs(z)^2, predicted class]. A adds reference/classifier binary disagreement;
K adds designated-control disagreement; P adds O=1-q if p>=0.5 and q otherwise.
q is `pathology_probability_AD`; K uses `control_probability_AD`.
`source_target` is the classifier error against `AD_label`.
Source heads use unweighted StandardScaler and L2 logistic C=0.1, with equal total weight per configuration
from `source_weights`. These are not the inputs/settings of the original equal-workload primary endpoint baseline.

CE adds population SD and 4f(1-f), where f is the proportion of all available rows for a participant predicting AD.
The latter is the normalized vote disagreement (0 for unanimous, 1 for evenly split predictions).
NACC has 58 configuration-source-fold rows per participant; in source and ADNI data the number of rows per participant is the number of available configurations (up to nine).
CEr percentile ranks use the cohort being scored, without labels, so this variant is a cohort-level procedure.

## Local refitting

`crossfit_scores` requires `participant_key`, `AD_label`, `model_probability_AD`, `pathology_probability_AD` and the
weight column (default `review_weight`, configuration-equal weights as in the reported analysis; `weight_col=None`
gives equal row weights). Seed 20261014 assigns participants to five folds at random (unstratified); all rows of a
participant stay in the same fold, and every training fold must contain both error classes.

## Review groups and weights

`Cohort` requires `participant_key`, `row_key`, `AD_label`, `model_probability_AD`, `review_weight`,
and grouping columns. ADNI groups by `configuration_id`; NACC by `configuration_id` and `source_fold`.
Each participant appears at most once per review group, label is constant per participant, and review weight is
constant within each group. Row keys must distinguish predictions. For center bootstrap, supply `center`, constant per participant.
Inputs and scores must be finite, nonempty and correctly aligned. Cohort evaluation pools weighted selected mass/errors;
its weight convention is distinct from the fitting-weight choice above.

`floor=True` selects floor(budget * group size) predictions in native unweighted counts, with row-key tie breaking.
`floor=False` in the weighted engine uses exact fractional selected mass, including a fractional boundary row.
Zero selected mass has undefined yield. `fixed_budget_yield` follows these conventions on a single group,
with input-position tie breaking. Prevalence reweighting requires both diagnosis classes in every bootstrap draw.
Bootstrap intervals use fixed scores and do not propagate fitting or model-selection uncertainty.

## Reports

`case_report` requires four severities and coefficients in `CORE_FEATURES` order.
The intercept plus contributions equals reference log-odds. The reference's contribution decomposition does not
identify a black-box causal mechanism. An empirical residual display threshold is not a clinical safety guarantee.
