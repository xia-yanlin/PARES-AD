# PARES-AD comparator and transport audit

Date: 2026-10-08 (Asia/Shanghai). This protocol is recorded before computing
the new comparisons, after ADNI/NACC outcomes and prior analyses were already
known. Every new result is post hoc/exploratory, not a fresh external confirmation.

## Scope and fixed inputs

1. Test simple competing explanations for incremental review information.
2. Describe cross-program disagreement differences on a common estimand and
   separate diagnosis composition from within-diagnosis joint behavior.
3. Prepare a usable human-audit study protocol and response/scoring templates.
   Actual recruitment and participant testing depend on collaborators and
   institution/data-use requirements. This computational analysis includes no
   recruitment or participant responses and supplies no evidence of human benefit.

No MRI model training, ROI/threshold search, new disease, new clinical anchor,
FW6, figures, manuscript edits or outcome-driven method selection. Existing
historical evidence files are read-only. Outputs go into a new dated directory.
NACC is the sole external resource. UID/date-description cohorts are evaluated
separately; their overlap and unknown ADNI overlap remain explicit.

## A. Four fixed error-risk scores

All scores predict errors of the same frozen MRI classifier. Common features:
absolute classifier logit, its square, and classifier predicted class (0.5).
Use the historical StandardScaler + LogisticRegression(C=0.1, lbfgs,
max_iter=2000), with equal total source configuration weight and no tuning.
The scaler follows the existing unweighted source-fitting implementation.

- C: confidence only, three features.
- A: C plus binary disagreement with the fixed AD reference, four features.
- K: C plus binary disagreement with the existing non-AD anatomical control,
  four features. This is the original control, not a newly selected one.
- P: C plus continuous reference opposition, four features: q when the MRI
  classifier predicts CN, 1-q when it predicts AD, where q is AD-reference
  probability. Direction alignment is determined only by model predictions,
  never the true diagnosis. This avoids an artificial same-sign linear q effect
  on errors in both predicted classes; it uses the full reference probability.

Train on existing ADNI out-of-domain development predictions and cross-fitted
references. For NACC use the original 12-input-masked source replay and
unchanged full-source deployment references; verify C/A reproduce the original
NACC scores to 1e-6. Replay checkpoints only when no saved prediction exists;
no MRI weights, calibrators or transforms are fitted or changed. ADNI evaluation
reuses saved 882-person confirmation MRI predictions and the historical
domain-held-out AD/control reference recipes. Source normalization/fits are
isolated from all confirmation/NACC labels. Reconstructed reference outputs
must reproduce historical ADNI RR before new endpoints are interpreted.

Fit and save source coefficients and external scores before opening evaluation
outcomes in the new evaluation stage. This is execution provenance only, not a
claim that prior outcome knowledge was removed. Save protocol, scripts, source
files, prediction hashes and access receipts. Participant rows remain private.

## A endpoints and inference

Main exploratory comparisons: A-C, A-K and A-P in review error yield at the
already used 5%, 10%, 20% budgets; retain all pairwise contrasts for transparency.
The 10% budget is an illustration, not a new confirmatory primary endpoint.
Review per configuration/source-fold for NACC; per configuration for ADNI
(one out-of-domain prediction per available configuration/person). Rank by
descending score, break ties by fixed private row key, floor the count.
NACC keeps original configuration/fold weights; ADNI weights each configuration
equally in aggregate and uses constant row weight within configuration.

Use 2,000 paired participant bootstrap draws, resampling all predictions of
each person together and reranking each draw. NACC additionally uses 2,000
paired center draws. No refitting in the evaluation bootstrap: uncertainty is
conditional on the fixed source score fits. Report percentile 95% intervals,
valid-draw counts, actual workloads, all-error/FP/FN capture, remaining error,
and all nine configuration directions. Nominal intervals across comparisons
and budgets are exploratory and not multiplicity-adjusted. Do not select the
best budget or method after evaluation. Report AUROC/AP/Brier/log-loss as
descriptive secondary error-risk metrics, not substituted primary endpoints.

Separately report diagnostic AUROC/AP/Brier/log-loss, sensitivity and specificity
of the original classifier, AD reference and non-AD reference on a matched
model-case population with matched weights. This is a comparison of diagnostic
outputs, not review yield or evidence of clinical utility/equivalence.

## B. Harmonized disagreement audit

Include ADNI confirmation, strict NACC and broad NACC. For each reference,
tabulate true diagnosis x classifier class x reference class (eight cells),
including both-correct/both-wrong and both directions of disagreement. This
distinguishes a reference correcting a classifier error from misleading it.

Use equal weight per participant, distributing that weight equally across
their available configurations and within each configuration across source
folds. NACC is equivalent to its historical weights up to a constant; ADNI
is a new descriptive estimand because configurations are not complete for
every person. Report observed configuration availability and all-configuration
strata; do not claim source-fold/measurement comparability is fully resolved.
Also reproduce the historical ADNI unweighted and NACC weighted RR receipts.

Standardize each cohort's conditional eight-cell distribution to ADNI
confirmation prevalence 342/882 (chosen from existing cohort definition, not
from performance). Report native and standardized RR, risk difference, flagged
fraction, absolute flagged/unflagged risk, error capture, and directional
reference win fractions. True labels are used only for this descriptive audit,
not a deployable selection rule.

For ADNI versus each NACC cohort, decompose native differences in RR and risk
difference by averaging both replacement orders (symmetric composition/within-
diagnosis decomposition): g(P,pi) combines class-conditional joint cells P with
AD prevalence pi. Composition = half[(g(A,piN)-g(A,piA)) +
(g(N,piN)-g(N,piA))]; conditional = half[(g(N,piA)-g(A,piA)) +
(g(N,piN)-g(A,piN))]. Verify exact sum to total difference. If a denominator
vanishes, report not estimable. These are descriptive arithmetic contributions,
not causal domain effects. No between-cohort bootstrap CI assumes independence
when ADNI-NACC overlap is unknown. Changes in deployment, diagnosis, processing
and reference fitting remain alternative explanations for the conditional part.

## Stopping and interpretation

Report all four scores, both linkage cohorts, all budgets and both error types.
If A does not exceed K/P, the study cannot claim a unique AD-disagreement
ranking advantage. A win against K alone would not exclude every generic
second-classifier explanation. Neither A nor P establishes molecular specificity.
No new ROI/control/score/threshold follows an unfavorable result. Standardized
results do not overturn the frozen external specificity failure. The original
ADNI primary review endpoint and later state/reason null result remain intact.

## Required verification and outputs

- Numeric tests against explicit bootstrap-copy selection including score ties,
  weighted ranking and small groups; source-label isolation; standardization
  and decomposition identities on synthetic data; no participant outputs in
  aggregate CSVs.
- Reproduce original NACC C/A scores and review yields and historical RR.
- Independently reconstruct representative real-data rankings and scalar
  cell arithmetic rather than relying solely on the production verifier.
- Save aggregate score comparisons, diagnostic comparisons, joint-state cells,
  standardized risks, decomposition, source provenance and a Chinese scientific
  audit explaining exactly which claims changed and which did not.
- Prepare human study instructions, randomization plan, independent rubric
  procedure and blank response/scoring forms. Mark it NOT RUN.
