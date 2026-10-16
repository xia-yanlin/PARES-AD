# Focused manuscript strengthening using the sole external cohort

These supplementary analyses use the existing NACC data. No additional external
dataset is included.

## Scope and timing

Written after the original NACC results and descriptive review-budget yields
were seen, before the clinical associations and new review-yield intervals
below are computed. All additions are explicitly post hoc/exploratory.
Original protocols, predictions, classifiers, references, risk scores,
thresholds, cohorts, and primary endpoints remain unchanged. NACC remains the
external evaluation population for the existing method, not an untouched test
of future method changes. Unknown ADNI overlap is disclosed, not treated as a
reason to discard all external evidence. No new dataset, PET hypothesis sweep,
longitudinal prediction task, retraining, recalibration, ROI search, or
threshold optimization is included.

## A. Clinical corroboration

Question: does the fixed AD anatomical reference carry clinical-severity
information beyond diagnosis, routine covariates, model output and the fixed
anatomical control? This directly extends the existing PACC evidence, without
asserting that a reference adjudicates diagnosis or proves molecular AD.

Main exploratory outcome: same-visit CDR sum of boxes (CDRSUM). Supporting
outcome: NACCMMSE, oriented as 30 minus score. Do not substitute another scale
after viewing associations. The matching visit is the nearest clinical visit
within the original +/-180-day window, with the original earlier-date tie
rule. Reconstruct the frozen scan key from the MRI table and require identical
diagnostic labels. Conflicting duplicate records for the selected visit are
excluded, not resolved by picking a favorable clinical value. No searching
later visits for a nonmissing score. Follow published dictionary valid codes.

Use one row per participant in each original linkage cohort. Complete-case
covariates are age, sex, education, diagnosis, clinical/MRI signed day gap,
center, form version, and the configuration/fold-weighted mean frozen model
logit. Compare base, base+AD-reference logit, and base+control logit using fixed
five-fold participant cross-fitting, balanced within diagnosis by deterministic
hashed-key order. Ridge alpha=1 (intercept unpenalized) is fixed, with numeric
features standardized within each training fold and categorical indicators.
Primary clinical contrast is cross-fitted R2(AD extension) minus R2(control
extension); also report both increments over base and raw R2. R2 uses the
outcome SST of the same complete-case sample for all models. No model selection.

Use 1,000 paired participant bootstrap draws with all copies of a participant
remaining in that participant's original fold. Refit the three clinical
regressions (including training scaling) in every bootstrap; do not refit the
frozen MRI models/references. Report missingness, sample composition, clinical
score floor/ceiling proportions and prediction-label agreement. Same-visit
clinical scales can inform the diagnosis and are corroboration, not an
independent diagnostic gold standard. CN/AD restriction limits spectrum.

## B. Fixed-workload review information

Extend the already specified 5%, 10%, 20% budgets without selecting a favorable
budget. Compare the two frozen source-trained error-risk scores. Reproduce
original point yields exactly. Within each configuration/source-fold select
floor(budget*N) predictions, sorting risk descending and row_key ascending.
For 2,000 paired participant bootstrap draws, resample participants jointly
across all cells, reselect the top budget in each draw, and recompute yields.
Repeat with center bootstrap as a dependence/heterogeneity sensitivity, using
floor(budget*resampled_N) in each center draw. Multiplicity counts are copies,
not independent repeated predictions. No NACC refitting.

Report yield difference and 95% percentile intervals, captured fraction of
all errors, false-positive and false-negative capture, remaining errors among
unreviewed predictions, and per-configuration direction. Review is a simulation
of error identification, not evidence that a reviewer corrects errors or saves
time. Intervals are exploratory and not multiplicity-adjusted. Budgets are
shown together; the 10% example illustrates workload rather than redefining a
primary endpoint. Both linkage cohorts are reported and not pooled.

## Manuscript deliverables and completion evidence

1. Add only the two evidence blocks above, with aggregate reproducible tables
   separate from frozen Table S5 and participant data kept private.
2. Tighten the main paper around anatomical evidence, coverage/abstention,
   replicated risk and practical complementary information. Move historical
   FW/probe development detail out of the abstract/main result sequence;
   preserve its honest boundary in supplementary context.
3. Preserve all material negative findings and scope qualifications. Integrate
   informative new results regardless of direction; no positive-only reporting.
4. Reconcile numerical text with aggregate outputs, verify frozen source hashes
   and cohort/clinical joins, review computational edge cases, and run the
   manuscript package validation. Record an evidence-based completion audit.
5. Completion means the strongest supported, coherent manuscript obtainable
   within this scoped work, not a guarantee of acceptance or claims of human
   benefit. Author identity, institutional ethics and funding remain author
   supplied submission metadata, not invented scientific results.
