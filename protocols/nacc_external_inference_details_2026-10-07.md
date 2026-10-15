# Statistical implementation details before external outcome access

Supplements the analysis freeze without changing populations, predictions,
primary endpoint or source rules. Written while Stage B runs, before Stage C.

Weighted AUROC/AUPRC use standard sample-weight definitions on model-cell
rows. For participant bootstrap, draw N participants with replacement and
multiply every row's fixed configuration/fold weight by that participant's
multiplicity. For center sensitivity, draw K centers with replacement and
preserve all original row weights within each center. Bootstrap percentile
limits use 2.5%/97.5%; record the number of nonfinite draws for each estimand.
Primary support also requires at least 95% valid draws.

For equal-workload binary-conflict review, within each configuration/source
fold select the same number of rows by ascending abs(probability-0.5) as the
number flagged by conflict. Compute weighted error capture and review yield
across cells. For learned risk models use floor(budget*N) rows per cell at
budgets 0.05, 0.10 and 0.20. Sort descending risk with hashed row_key as the
stable tie breaker. Do not use outcomes to sort or break ties.

Behavioral R2 is a descriptive weighted residual-sum-of-squares statistic on
model logits. The source quantile is an empirical calibration bound, not an
external guarantee. State-specific risks compare disease-only covered
PATHOLOGY_CONFLICT and UNEXPLAINED_ABSTAIN with covered PATHOLOGY_ALIGNED;
the primary binary-disagreement endpoint includes all rows.

Primary and anatomical-control bootstrap intervals are computed with both
participant and center sampling. Paired secondary bootstrap intervals are
computed for the two learned confidence risk models' AUROC/AUPRC/Brier/log
loss differences. Fixed-budget capture and state risks are descriptive in
this run. Stratified summaries include predicted class and true clinical
class as diagnostics; no new favorable subgroup becomes a primary claim.
