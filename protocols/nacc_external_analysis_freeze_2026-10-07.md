# PARES-AD NACC75 external evaluation: prospective local analysis freeze

This protocol is written after schema, diagnosis-count and source-only
missing-input assessment, but before NACC model predictions, errors,
reference accuracy, conflict associations or risk-ranking results. It is a
timestamped local freeze, not a public preregistration.

## Question and population

Does prespecified AD structural counterevidence identify an error-prone state
in a different AD/MRI program under a disclosed missing-input deployment
condition, and how do explanation coverage and confidence comparators behave?
This is not the terminated prospective model-selection question.

Evaluate both received-data receipts without changing their clinical rules:

- The broad **provisional linkage cohort** uses unique participant, center,
  scan-date, selected T1w and identical sequence-description linkage. Its
  current receipt is 3,797 participants (491 AD, 3,306 CN).
- The **exact-UID linkage sensitivity cohort** uses participant plus exact
  T1-series UID, currently 145 participants (32 AD, 113 CN). It has its own
  earliest eligible scan; do not assume it is a row-identical subset of the
  broader cohort. Compute and report the overlap at participant level.

Both use the previously frozen QC/completeness and nearest-any-status visit
within 180 days rules. Receipt counts must reproduce. No adding a wider
clinical window, later eligible diagnosis or alternative ROI after outcomes.
The broad cohort does not earn verified image-level linkage from a favorable
result. Program-level external evaluation cannot be called proven
participant-independent validation while cross-program overlap is unknown.

## Deployed black boxes and references

Use all 290 saved jobs in the original manifest, all five seeds per each of
58 available configuration/source-fold cells, representing nine configurations
and three families. Each NACC participant is evaluated in every available
cell. Average five seed probabilities within a cell only. A source fold is
not a NACC acquisition domain, independent cohort, or independent participant.
Do not choose a favorable source fold, architecture, configuration or seed.

Apply the saved non-ComBat preprocessors and Platt calibration unchanged.
Set only the 12 structurally unavailable fields to NaN so the existing
training-median handler applies. Other required observed inputs must be
positive and finite. Do not fit any preprocessing or calibration on NACC.

Fit the original monotone four-concept AD reference and original
equal-complexity low-AD-prior control once on the original ADNI baseline
development cohort, with their unchanged definitions, training-only
normalization and L2 penalty 0.1. Use the complete ADNI source cohort as the
final training reference; use its existing domain-held-out predictions for
source risk-model training. No NACC outcome enters either reference.
Retain exact signed contributions and verify their logit reconstruction.

The October 5 closest-performance reference is an exploratory extension
after repeated ADNI confirmation access and not an exact matched comparator.
It will not be used to select or replace the frozen primary anatomical
control in NACC. Existing findings and limitations remain in the manuscript.

## Behavioral coverage in a new program

Retain the disease-only Ridge(alpha=1) source-trained reconstruction as an
externally testable configuration-level behavioral approximation. Estimate
its absolute residual bound by leaving source acquisition domains out, using
the original empirical 90% quantile formula. Fit its final disease-only
surrogate using source out-of-fold black-box logits and corresponding
cross-fitted concept severities for each configuration, using the exact NACC
missing-input pattern on source predictions. Deploy using the final
source-trained concept transforms. Report the resulting external empirical
coverage and R2. Do not claim nominal conformal coverage under program shift.

NACC has no ADNI phase. Do not insert an ADNI phase or interpret a substituted
phase coefficient. Mark acquisition attribution as unavailable for this
external program. External covered/conflict/aligned and residual-abstain
outputs are the **disease-only deployment projection**, not validation of the
original complete five-state acquisition-dependent contract. The latter
remains not established externally. Report the ungated binary structural
disagreement endpoint separately from the coverage-gated states.

## Weighting and primary evidence

Give each configuration equal total weight, and source-fold cells equal
weight within configuration: a cell row has weight 1/(number of source folds
available for its configuration). Every participant then contributes equal
total weight across nine configurations. Report raw counts and unique people
separately; repeated model predictions never increase the participant n.

Primary: ratio of weighted model-error risk under binary AD-reference
disagreement to agreement at the unchanged 0.5 class threshold. Report risk
difference and absolute error rates too. Predefined replication criteria:
RR >= 2, participant-cluster 95% CI lower > 1, positive direction in at least
7/9 configurations and all 3 families. All criteria and all configurations
must be reported. If subgroup denominators vanish, mark not estimable.

Key falsification: AD-minus-control difference in conflict risk ratios with
paired participant-cluster interval. A disease-pattern superiority claim
requires this interval entirely above zero; otherwise retain the generic
disagreement alternative. This test follows primary replication in the claim
hierarchy and cannot rescue a failed primary result.

## Secondary and bounded endpoints

- Reference and each classifier discrimination (AUROC/AUPRC), Brier score,
  prevalence, class prediction fraction and class-specific errors.
- Disease-only behavior reconstruction R2, empirical coverage, state
  prevalence and coverage-gated conflict/abstain error risks.
- A confidence-only error model and confidence-plus-binary-conflict model:
  identical source-only StandardScaler + LogisticRegression(C=0.1), features
  abs(logit), abs(logit)^2, predicted class; the second adds conflict. Use
  the original source domain-held-out AD reference conflict and masked
  source black-box probabilities. Weight source rows to equalize each
  configuration. Fit no NACC risk model. Report paired external error-risk
  AUROC/AUPRC/Brier/log-loss changes. These are secondary, not clinical benefit.
- Equal workload confidence-distance versus binary conflict review, plus
  fixed 5%, 10%, 20% budgets for the two learned risk scores. Ties use a
  stable local hashed participant key and source-cell order, never labels.
- Report class strata, center variation, and source-configuration/family
  directions. Clinical labels define errors but not model features or states.
- Report identical endpoints in the exact-UID cohort without using it to
  tune the broad-cohort rules. A broad-positive but UID-inconclusive result
  remains linkage-limited; a UID contradiction requires explicit discussion.

## Inference, failure and data separation

Use 2,000 deterministic paired bootstrap samples at participant level
(seed 20261007). Resample whole participants with all model cells together.
For primary RR and AD-minus-control contrast also report a center-cluster
bootstrap sensitivity, preserving all participants within each resampled
center. Confidence intervals on secondary endpoints are descriptive, without
claiming a familywise-confirmed discovery. Never treat folds/seeds as
independent bootstrap units. Report invalid bootstrap draws.

Stage A creates and hashes an outcome-free local feature cohort with opaque
row keys and a separate outcome table; no model endpoints are computed.
Stage B freezes source models/rules and writes outcome-free predictions and
states with hashes. Stage C verifies those hashes, then joins outcomes and
computes every frozen endpoint. Participant data and predictions remain in
the Git-ignored private output directory. Only aggregate reports enter the
manuscript. Stop on incompatible keys, prediction incompleteness, source
integrity failures, unexpected model NaNs or exactness failures.

An external negative result changes claim strength and scope. It does not
authorize another threshold, cohort window, model family or endpoint search.
No NACC molecular endpoint or human benefit experiment is part of this
evaluation. Those claims remain unsupported unless separately designed.
