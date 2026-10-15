# PARES-AD independent participant confirmation freeze

**Freeze ID:** `PARES-AD-INDEPENDENT-CONFIRM-v1`  
**Frozen date:** 2026-07-23  
**Status:** `FROZEN_BEFORE_CONFIRMATION_INFERENCE`  
**Scope:** one low-cost, conclusion-changing confirmation of the AD/MRI-specific
PARES-AD review trigger. This is not FW6, a new concept audit, an ablation study,
or a cross-disease generalization study.

## 1. Practical problem and proposed solution

An AD-versus-CN MRI classifier can be confidently wrong. Confidence alone does
not tell a reviewer whether the prediction agrees with the canonical structural
pattern of AD. PARES-AD supplies a second, pathology-grounded view based on four
prespecified FreeSurfer summaries:

1. bilateral hippocampal atrophy, adjusted for eTIV;
2. bilateral entorhinal thinning;
3. bilateral ventricular enlargement, adjusted for eTIV;
4. a prespecified temporoparietal cortical-thickness signature.

For each acquisition domain, a transparent monotone logistic reference is fit
only on baseline participants outside that domain. Each of its four coefficients
is constrained to be nonnegative after orienting the features so that larger
values mean greater AD-like severity. Its signed log-odds contributions sum
exactly to the reference prediction.

The operational rule is deliberately narrow:

> Send a case for review when the frozen black-box AD/CN classification and the
> frozen monotone AD-pathology reference classification disagree.

This `PATHOLOGY_CONFLICT` flag is not a diagnosis, causal mechanism, or claim
that the black box internally uses the four features. It is a reproducible
second-reader disagreement signal.

## 2. Why this confirmation is necessary

The development evidence used 650 participants from the original 729-person
baseline cohort. Across 5,184 model-case predictions from nine configurations
and three neural families:

- pathology-conflict error rate was 46.50% versus 5.82% when aligned;
- subject-clustered error-risk ratio was 7.99 (95% interval 4.81–12.59);
- at a review budget equal to each configuration's conflict count, conflict
  review captured 133 errors versus 106 under the already cross-domain,
  confidence-only error-risk ranking, an absolute review-yield gain of 9.44
  percentage points;
- conflict review had a higher error capture count in eight of nine
  configurations.

Those figures are development evidence. They motivated the frozen operational
rule and cannot establish independent utility.

The confirmation changes the scientific claim only if the fixed rule works on
participants who were absent from every baseline training, calibration, outer
test, pathology-reference, and error-risk-model fit.

## 3. Data receipt and outcome-blind feasibility result

No checkpoint inference on the confirmation cohort and no confirmation
explanation-versus-error association was inspected before this freeze.

Read-only feasibility checks found:

- 10,526 FreeSurfer scan rows from 2,964 participants;
- 1,995 participants with repeated scans;
- 1,981 candidate rows from 882 participants absent from the 729-person
  baseline cohort, with an exact visit-level AD/CN diagnosis and an eligible
  acquisition domain;
- all 1,981 candidate rows had finite, strictly positive values for all 163
  frozen raw model inputs (94 volumes, 68 cortical thicknesses, and eTIV);
- selecting the earliest eligible scan yielded exactly one scan for each of 882
  independent participants, with no same-date ties;
- the frozen confirmation cohort contains 342 AD and 540 CN participants.

Frozen domain counts:

| Outer acquisition domain | Total | AD | CN |
|---|---:|---:|---:|
| ADNI1_GE | 112 | 68 | 44 |
| ADNI1_Philips | 64 | 38 | 26 |
| ADNI1_Siemens | 184 | 123 | 61 |
| ADNI3_GE | 85 | 18 | 67 |
| ADNI3_Philips | 69 | 18 | 51 |
| ADNI3_Siemens | 277 | 61 | 216 |
| ADNI4_Siemens | 91 | 16 | 75 |
| **Total** | **882** | **342** | **540** |

No participant identifier may appear in a confirmation output.

## 4. Frozen cohort construction

### 4.1 Diagnosis linkage

1. Convert `RID` to a canonical integer string internally.
2. Normalize `VISCODE2` by trimming whitespace and converting to lowercase.
3. Group DXSUM by canonical `RID` and normalized `VISCODE2`.
4. Retain a group only if all nonmissing `DIAGNOSIS` values have one unique
   value and that value is `1` (CN) or `3` (AD).
5. Join MRI and DXSUM only on exact canonical `RID` plus normalized `VISCODE2`.
6. Do not use nearest-date, phase-only, diagnosis-history interpolation, or
   post hoc relabeling.

### 4.2 Participant independence

Exclude every participant whose canonical `RID` occurs in
`baseline_cohort_with_splits.csv`. This excludes all participants used for the
original model fitting, calibration, outer evaluation, pathology-reference
development, and PARES-AD rule development.

### 4.3 Acquisition normalization

Use the previously frozen fail-closed rules:

- phase: trimmed uppercase, allowed `ADNI1`, `ADNI3`, `ADNI4`;
- manufacturer: any `SIEMENS` substring to `Siemens`, any `PHILIPS` substring
  to `Philips`, and `GE`, `GE ` prefix, or `GE MEDICAL` substring to `GE`;
- field strength: numeric 1.5 to `1.5T`, numeric 3.0 to `3.0T`;
- acquisition domain: canonical phase plus `_` plus canonical manufacturer.

Retain only the seven domains already present in the frozen FW5 outer
evaluation. Unknown values fail closed.

### 4.4 MRI QC and one-scan rule

Require a valid `PHC_SCANDATE`, positive eTIV, and finite strictly positive
values for every raw/non-ComBat volume and cortical-thickness input in
`noncombat_roi_mapping_v2_1.json`. No ComBat feature is used.

For each remaining participant, select the earliest eligible scan date. If more
than one eligible scan occurs on that earliest date, exclude that participant
rather than choose by an identifier or outcome.

No repeated scan from a confirmation participant enters the primary analysis.

## 5. Frozen model and reference inference

### 5.1 Black-box models

Use all 290 completed FW5 checkpoints exactly as stored:

- nine configurations from flat MLP, GAT, and no-edge ROI encoder families;
- five seeds for every eligible outer-domain/configuration unit;
- each checkpoint is applied only to new participants in its own held-out
  acquisition domain;
- each checkpoint uses its stored `preprocessor.npz`, graph adjacency where
  applicable, and stored Platt coefficient/intercept;
- average the five calibrated AD probabilities within each
  domain/configuration/case;
- classify AD at probability `>= 0.5`.

No model is retrained, recalibrated, selected, or dropped based on confirmation
performance. Configuration/domain units absent from the original frozen FW5
manifest remain absent.

### 5.2 Transparent pathology reference

For each outer domain:

1. use only baseline participants outside that domain as the fitting set;
2. derive the four oriented severity variables using the existing
   training-local transformations;
3. fit the same monotone nonnegative logistic reference with L2 penalty 0.1;
4. apply it unchanged to confirmation participants in that domain;
5. classify AD at probability `>= 0.5`.

The pathology reference is refit only to reproduce the already frozen
domain-held-out training operation. Confirmation labels are never used to fit
or recalibrate it.

### 5.3 Confidence-only comparator

For every model configuration and outer domain, reproduce the already specified
confidence-only error-risk model:

- training cases: original PARES-AD development predictions for all other
  acquisition domains;
- outcome: black-box classification error;
- predictors: calibrated AD probability, absolute distance from 0.5, and
  predicted class;
- estimator: `StandardScaler` followed by logistic regression with `C=1.0`,
  maximum 2,000 iterations, and random seed 20,260,723.

Apply this fitted model to confirmation cases from the held-out domain. No
confirmation error is used for training or calibration.

## 6. Frozen endpoints

### 6.1 Unique primary endpoint

**Equal-budget review error-yield gain**

Within each model configuration:

1. let `k` be the number of confirmation predictions flagged
   `PATHOLOGY_CONFLICT`;
2. pathology review sends those `k` predictions to review;
3. confidence review sends the `k` predictions with the highest frozen
   confidence-only predicted error risk to review;
4. pool captured errors and review counts across the nine configurations;
5. compute the absolute difference in captured-error yield:

`errors captured / predictions reviewed` under pathology-conflict review minus
the same quantity under confidence-only review.

Subject-cluster bootstrap resamples confirmation participants with replacement
2,000 times. On every replicate, `k` is recomputed within configuration and the
confidence review set is re-ranked within configuration. The percentile 95%
interval is reported.

This endpoint answers the practical question: at the same review workload, does
the AD-specific explanation identify more black-box mistakes than model
confidence alone?

### 6.2 Required corroborating endpoints

1. Error-risk ratio for pathology-conflict versus pathology-aligned
   predictions, with a 2,000-replicate subject-cluster bootstrap interval.
2. Direction of the conflict-versus-aligned error-rate difference in each of
   the nine model configurations and each of the three model families.
3. Review burden, number of reviewed predictions, and number of unique
   participants represented by a conflict flag.
4. Context only: pooled and domain-specific AUROC, AUPRC, Brier score, and
   calibration slope for the black boxes and transparent pathology reference.

Only the equal-budget review-yield gain is the primary endpoint.

## 7. Frozen success, failure, and stop gates

The independent confirmation is a **success** only if all conditions hold:

1. all 882 frozen participants are processed, all 290 checkpoint artifacts pass
   their existing completion/hash checks, and all five seeds are present for
   every manifest unit;
2. conflict review burden is between 3% and 15% of model-case predictions and at
   least 50 unique participants receive at least one conflict flag;
3. primary equal-budget review-yield gain is at least **+5.0 absolute
   percentage points** and its subject-cluster bootstrap 95% lower bound is
   greater than 0;
4. pathology-conflict versus pathology-aligned error-risk ratio is at least
   **2.0** and its subject-cluster bootstrap 95% lower bound is greater than 1;
5. conflict error rate is higher than aligned error rate in at least **7 of 9**
   configurations and in all three model families.

If any condition fails, stop this PARES-AD review-trigger route. Do not change
the threshold, confidence comparator, scan selection, ROI set, model set,
bootstrap unit, or primary endpoint. A failure may still be recorded as an
internal negative result but is not to be packaged automatically as a paper
contribution.

Passing these gates permits the claim that an AD-structural-pathology
disagreement flag improves error-focused review triage over confidence alone in
an independent ADNI participant set. It does not by itself permit claims of
external-cohort generalization, clinician benefit, causal mechanism, or
cross-disease validity.

## 8. Runtime and resource stop

Expected runtime is 3–10 minutes on CPU: 290 small checkpoint inferences over
at most 882 rows plus 2,000 low-dimensional subject-cluster bootstrap
replicates. If a single run is projected to exceed 30 minutes, stop before
launching or terminate at the next safe checkpoint and report the reason.

The run may create only:

1. one confirmation result CSV;
2. one machine-readable summary JSON;
3. one concise confirmation report.

No participant-level identifier mapping, intermediate tensor dump, new model,
new dataset, large resampling file, or manuscript package is permitted.

## 9. Frozen source receipts

- Raw FreeSurfer table SHA-256:
  `5093ebfcfd1330eb88390aafe3c3bc9470063e537864d11824363f828c0cddd8`
- DXSUM SHA-256:
  `d7c484fcaf56d4f7d83a4f862a805bfafc4146ccd91e3baab02e7bb8b455d42f`
- Baseline cohort with splits SHA-256:
  `246db0323d0ba9e0fdb3987bc0b15ea834c06943fa083df7caedd03e7faa5613`
- Baseline cohort with concepts SHA-256:
  `ffbe5879de554abdacd29fa0a75685311aed7c839876b2a75dff16c7498c4ce5`
- Frozen domain assignments SHA-256:
  `52813725ce19cdf1452dcee71848292df467529e200c5bf14fe77ab8e91ede4c`
- FW5 manifest SHA-256:
  `7a6c618f7985b430b6a74e3ab48a66c69a18e183b441cc6c48375fef66b5e4d9`
- Model registry SHA-256:
  `10c5d5c38ca1d07a679218204ec1bb4a31d31eeeb3ecbdcc00a728bb53aa90c7`
- Raw ROI mapping SHA-256:
  `8ed9e565afb3f46e6bc4ec1a4e9d088d470184842a2317d43e6b8580dc64c3db`
- Development PARES-AD case explanations SHA-256:
  `a7ad3530154e71d0789188e498b60479ddc6af612d46e1840979f769b4d7acfa`

## 10. Probability audit at freeze

These are planning estimates, not results:

- correct construction of the 882-participant confirmation cohort: >95%;
- successful replay of all frozen checkpoints without retraining: 90–97%;
- replication of conflict error-risk enrichment: 65–80%;
- passing the stricter equal-budget review-yield endpoint: 55–70%;
- if all gates pass, reaching a defensible single-disease HISS-level empirical
  contribution after scientific audit: 60–75%;
- unconditional probability of reaching that level from this frozen run:
  approximately 40–55%.

