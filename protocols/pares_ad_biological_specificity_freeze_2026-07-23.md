# PARES-AD biological specificity and confidence-blind-spot freeze

> **Correction dated 2026-10-09:** The historical PACC interval and positive
> clinical-severity conclusion in the execution record below were superseded.
> The corrected AD-minus-control estimate is 0.0184151, with a 95% interval
> of [-0.0025473, 0.0381420]; clinical-severity specificity is not established.
> See `pares_ad_content_correction_2026-10-09.md` and Supplementary Section S4.
> The historical record is retained to document the original analysis.

**Freeze ID:** `PARES-AD-BIOSPECIFICITY-v1`  
**Freeze date:** 2026-07-23  
**Status:** `COMPLETED_AFTER_CONTINUATION_AMENDMENT`  
**Research role:** one deepening experiment inside the unified PARES-AD
evidence chain; not a new research direction

### Interpretive clarification before FTP receipt

This clarification changes no cohort, endpoint, threshold, control, covariate,
bootstrap, or stopping rule. The conjunctive success gate in Section 9 governs
only the proposed upgrade from an anatomical-concordance warning to
AD-biologically specific counterevidence. It cannot invalidate the completed
findings that the transparent reference is exact and monotone, held-out
behavioral reconstruction requires abstention, or pathology-conflict error
enrichment replicated in independent participants. It also cannot reverse the
failed equal-workload comparison with confidence.

Accordingly, this experiment is neither a replacement for the PARES-AD system
nor a rescue attempt for a failed endpoint. It is the single falsification test
of the strongest remaining alternative explanation: ordinary disagreement
between two label-trained classifiers. Failure contracts the interpretation
and stops further same-cohort searching; it does not rewrite completed results
as nonexistent.

### Execution receipt

The outcome-blind FTP receipt found 295 eligible independent participants (64
AD, 231 CN) across four acquisition domains, with median absolute FTP-to-MRI
gap 20 days. Only 31 unique eligible participants had one or more frozen
AD-reference pathology conflicts, below the required 40. The molecular-
specificity analysis therefore stopped before any tau value was exported or
analyzed. Amyloid and PACC values were not accessed as rescue outcomes.

A 100-replicate timing pilot for the non-molecular adjusted-error analysis
projected 2,595 seconds for the frozen 2,000-replicate analysis, exceeding the
1,800-second cap. It produced no formal endpoint estimate or output artifact;
the full adjusted/control/high-confidence analysis was not run. The timing
failure is an execution stop, not a scientific result.

### Continuation amendment

Before any PET/PACC value was accessed, the count and runtime planning gates
were reclassified as feasibility and resource-management warnings.
The analysis is therefore resumed with all cohorts, anchors, ROIs, linkage
windows, endpoints, covariates, comparators, and bootstrap definitions
unchanged. The amendment changes interpretation only:

- 31 FTP-linked conflict participants is an imprecision warning, not an
  automatic failure; estimates and participant-cluster intervals must be
  reported without claiming adequate power;
- the 30-minute projection is a compute-management warning, not a scientific
  failure; implementation may be optimized and run in stages, with probability
  updates before each costly node;
- the original material-effect thresholds remain the criteria for strong
  support, but failure of a threshold contracts the claim rather than erasing
  other completed evidence;
- no outcome-driven ROI, anchor, window, covariate, model subset, or endpoint
  change is permitted.

### Final execution result

The optimized implementation matched the original adjusted-logistic solver to
a maximum relative odds-ratio difference of 0.000175% over matched bootstrap
samples and completed the full non-molecular analysis in 23.2 seconds.

- AD conflict versus equal-complexity non-AD anatomical control passed: risk-
  ratio difference 2.67, 95% participant-cluster interval 1.69–4.02.
- Conditional error association passed: adjusted odds ratio 4.64, 95%
  interval 2.30–10.55.
- PACC clinical-severity specificity passed: AD-minus-control cross-fitted R²
  0.0184, 95% interval 0.0042–0.0249.
- FTP molecular specificity did not pass: all four frozen tau gates failed.
- The high-confidence subset remained imprecise: risk ratio 3.45, 95%
  interval 0.62–7.86.

The final status is anatomical and clinical-severity specificity supported
within ADNI, molecular specificity not supported. This result does not imply
superior confidence ranking, clinician benefit, or external-program transport.

## 1. Fixed framework claim

PARES-AD is an AD/MRI evidence audit with four linked layers:

1. measurement context;
2. exact AD-anatomical support and counterevidence;
3. held-out functional fidelity with unexplained-residual abstention;
4. explicit decision-value and failure boundaries.

The existing independently confirmed result is that a disagreement between a
black-box AD/CN prediction and the monotone four-feature AD reference identifies
a high-error state. The unresolved alternative explanation is that this is
ordinary disagreement between two label-trained classifiers rather than
AD-specific medical counterevidence.

This experiment tests that alternative directly. It does not revisit the failed
claim that pathology conflict catches more errors than confidence at equal
review workload.

## 2. Theory linking the evidence layers

### 2.1 Measurement error precedes explanation

An explanation can be faithful to a model while the model input is an unstable
measurement. The existing scanner-switch analysis therefore supplies the
measurement premise: acquisition context must be shown, but it is not itself
the PARES-AD main endpoint.

### 2.2 A transparent reference is counterevidence, not ground truth

The AD reference is an inherently interpretable monotone model. Its four signed
log-odds contributions sum exactly and have prespecified disease directions.
This gives inspectable evidence, but clinical-label prediction alone cannot
establish biological specificity.

### 2.3 Functional reconstruction bounds what may be explained

The black-box decomposition is a held-out functional surrogate, not a causal
account of internal neurons. Residual-bound failure therefore triggers
abstention. The biological-specificity analysis validates the transparent
reference and conflict state; it does not turn the surrogate into a causal
mechanism.

### 2.4 AD biomarkers have non-exchangeable roles

The NIA-AA `AT(N)` framework and the 2024 Alzheimer's Association biological
criteria distinguish amyloid (`A`), tau (`T`), and neurodegeneration (`N`). MRI
atrophy is an `N` measure: clinically relevant but not etiologically specific.
Amyloid establishes AD pathophysiology but is not a direct regional atrophy
truth. Tau burden is used for biological staging and is more proximal to the
regional neurodegenerative pattern. Cognitive performance is a clinical
phenotype influenced by AD and non-AD factors.

Therefore:

- **primary molecular anchor:** FTP meta-temporal tau PET;
- **etiologic corroboration only:** amyloid PET status;
- **clinical corroboration only:** PACC;
- neither amyloid nor PACC may rescue failure of the tau primary gate.

Primary framework references:

- Jack et al. *NIA-AA Research Framework: Toward a biological definition of
  Alzheimer's disease*. Alzheimer's & Dementia. 2018;14:535–562.
  doi:10.1016/j.jalz.2018.02.018.
- Jack et al. *Revised criteria for diagnosis and staging of Alzheimer's
  disease: Alzheimer's Association Workgroup*. Alzheimer's & Dementia. 2024.
  doi:10.1002/alz.13859.
- Guo, Ustun, and Hullman. *Explanations are a Means to an End:
  Decision-Theoretic Explanation Evaluation*. ICML 2026.

## 3. Data roles and access history

### 3.1 Existing development and confirmation evidence

The original 729-person baseline cohort, FW5 outer predictions, PARES-AD
development explanations, and the 882-participant independent confirmation are
already accessed. Model errors and the failure of equal-budget review yield are
known.

Consequently, conditional-error and high-confidence analyses in this protocol
are prespecified deepening analyses, not fresh prospective confirmation.

### 3.2 Still-unaccessed orthogonal outcomes

Before this freeze, only RDA dictionaries, identifiers, dates, QC flags,
tracer names, and missingness indicators were read. No tau SUVR, amyloid status,
PACC value, or relationship between any anchor and PARES-AD was inspected.

Outcome-blind coverage receipt for the 882 independent participants:

- high-quality amyloid PET within ±730 days: 366 participants, six acquisition
  domains, median absolute gap 21 days;
- high-quality tau PET within ±365 days: 308 participants, four acquisition
  domains, median absolute gap 21 days;
- nonmissing PACC within ±180 days: 876 participants, seven acquisition domains,
  median absolute gap 6 days.

The tau table contains FTP, MK6240, and PI2620 and explicitly warns against
cross-tracer SUVR comparison. Only FTP is eligible for the primary tau analysis.

## 4. Frozen participant and model cases

Use exactly the 882-participant independent cohort and exactly the available
7,091 five-seed, outer-domain model-case predictions from
`PARES-AD-INDEPENDENT-CONFIRM-v1`.

Reconstruct these predictions from the 290 hash-verified FW5 checkpoints using
their stored preprocessing, adjacency, and Platt calibration. Do not retrain,
recalibrate, select, or drop models based on this experiment.

The four-feature AD pathology reference is reproduced exactly as frozen:

- bilateral hippocampal atrophy adjusted for eTIV;
- bilateral entorhinal thinning;
- bilateral ventricular enlargement adjusted for eTIV;
- prespecified bilateral temporoparietal thinning.

Each reference is trained only on baseline participants outside the evaluation
acquisition domain.

## 5. Frozen equal-complexity anatomical falsification control

Fit an identically regularized monotone logistic reference using four
low-prior/general-neurodegeneration features with matching transformation types:

1. bilateral caudate volume, log-eTIV residualized and oriented as atrophy;
2. bilateral pallidum volume, log-eTIV residualized and oriented as atrophy;
3. bilateral pericalcarine thickness, oriented as thinning;
4. bilateral precentral thickness, oriented as thinning.

The same domain-held-out training rows, L2 penalty `0.1`, nonnegative
coefficients, and probability threshold `0.5` are used. No control region may be
replaced after outcomes are read.

This is a stringent specificity control, not a claim that these regions are
unaffected by age or AD. If it performs like the AD reference, the AD-specific
interpretation fails.

## 6. Frozen anchor linkage

### 6.1 FTP tau primary cohort

Source: `UCBERKELEY_TAU_6MM.rda`.

- require `TRACER == "FTP"`;
- require `qc_flag == 2`;
- require finite positive `META_TEMPORAL_SUVR`;
- choose the scan with minimum absolute date difference from the MRI within
  365 days;
- on a distance tie, prefer PET on or before MRI, then the earlier PET date;
- require at least 150 participants, at least 35 AD and 75 CN participants,
  at least four acquisition domains, and at least 40 unique participants with
  one or more AD-reference pathology conflicts;
- if any count fails, tau specificity is not sufficiently estimable and the
  molecular-specificity claim stops before fitting outcome models.

Use `log(META_TEMPORAL_SUVR)` because the outcome is positive and right-skew is
biologically plausible. Standardize using the eligible FTP sample mean and
standard deviation only after the cohort is fixed.

### 6.2 Amyloid corroboration

Source: `UCBERKELEY_AMY_6MM.rda`.

- require `qc_flag == 2` and nonmissing `AMYLOID_STATUS`;
- nearest scan within 730 days with the same tie rules;
- tracer-specific SUVRs are not compared; only the official binary status is
  used;
- report as etiologic corroboration only.

### 6.3 PACC corroboration

Source: `PACC.rda`.

- require finite `mPACCtrailsB`;
- nearest `PACC_VISDATE` within 180 days with the same tie rules;
- higher scores are better; analyses orient `-PACC` as greater clinical burden;
- report as clinical corroboration only.

No CSF, alternative PET ROI, alternative cognitive score, longitudinal change,
or threshold search is permitted.

## 7. Covariates and outcome-blind residualization

Participant covariates are fixed before anchor modeling:

- MRI age (`PHC_Age_T1`);
- sex from the earliest valid PTDEMOG record;
- education from the earliest valid nonnegative PTDEMOG record, with a missing
  indicator and median imputation if needed;
- clinical diagnosis (`CN` or `AD`);
- MRI phase, manufacturer, and field strength;
- absolute anchor-to-MRI date gap.

For tau and PACC separately, generate participant-level five-fold cross-fitted
baseline residuals with seed `20260723`. The baseline ordinary least-squares
model contains only the covariates above. Tau/PACC outcomes from an evaluation
fold are never used to fit its residualization model.

For tau, positive residual means more tau than expected from diagnosis and the
prespecified covariates. For PACC, multiply the residual by `-1`, so positive
means worse cognition than expected.

## 8. Frozen endpoints

### 8.1 Primary endpoint A — tau validity of the AD reference

At the participant level, compare cross-fitted squared-error reduction for
predicting standardized FTP log-tau residual using one additional frozen score:

- AD pathology logit;
- control-reference logit.

Scores are each standardized within the fixed FTP cohort. Fit the additional
coefficient only in the training folds and predict the held-out fold.

Report incremental cross-fitted R² over the covariate-only residual prediction
of zero:

- `delta_R2_AD`;
- `delta_R2_control`;
- `delta_R2_specificity = delta_R2_AD - delta_R2_control`.

Participant bootstrap with 2,000 replicates repeats the fixed-fold prediction
comparison without changing definitions.

### 8.2 Primary endpoint B — molecular adjudication of model conflict

For every tau-linked model-case pathology conflict, define:

`direction = +1` when AD reference predicts AD and black box predicts CN;

`direction = -1` when AD reference predicts CN and black box predicts AD.

The signed tau adjudication score is:

`direction * standardized cross-fitted tau residual`.

A positive value means independent tau burden leans toward the AD reference
rather than the black box. Compute the same score for control-reference
conflicts. Use a 2,000-replicate participant-cluster bootstrap.

### 8.3 Required specificity controls

1. Compare AD-reference and control-reference signed tau adjudication.
2. On all 882 participants, compare the model-error risk enrichment of AD
   conflict and control conflict with identical subject-cluster bootstrap.
3. Fit a descriptive adjusted logistic model for model error using conflict,
   frozen confidence-error risk, predicted class, age, sex, education, phase,
   manufacturer, field, model configuration, and outer domain. Do not include
   true diagnosis together with predicted class because that would determine
   the error outcome. Bootstrap participants for the conflict coefficient.

The adjusted logistic result is conditional evidence, not a new prospective
prediction model or a decision-improvement endpoint.

### 8.4 Confidence blind spot

Define high confidence before analysis as calibrated probability `<=0.10` or
`>=0.90`. Within this fixed subset, compare pathology-conflict and
pathology-aligned error risk with subject-cluster bootstrap and report the
fraction of all high-confidence errors carrying a conflict flag.

This tests whether PARES-AD provides inspectable counterevidence for predictions
that confidence alone presents as safe. It does not compare ranking policies.

### 8.5 Non-rescuing corroboration

- Amyloid: association of AD pathology logit with official amyloid status after
  covariates, compared with control logit.
- PACC: the same cross-fitted score and conflict-adjudication analyses as tau,
  using oriented clinical-burden residual.

These results strengthen interpretation only if tau primary gates pass. They
cannot rescue tau failure.

## 9. Conjunctive success and stop gates

The new AD-biological-specificity link is supported only if **all** hold:

1. FTP feasibility counts in Section 6.1 pass;
2. `delta_R2_AD >= 0.02`;
3. `delta_R2_specificity >= 0.01` and its 95% bootstrap lower bound is greater
   than zero;
4. mean AD-reference signed tau adjudication is at least `0.20 SD` and its 95%
   bootstrap lower bound is greater than zero;
5. AD-reference minus control-reference signed tau adjudication is at least
   `0.10 SD` and its 95% bootstrap lower bound is greater than zero;
6. the adjusted pathology-conflict error odds ratio is at least `2.0` with 95%
   lower bound above one;
7. among high-confidence predictions, pathology-conflict error risk ratio is at
   least `2.0`, its 95% lower bound exceeds one, and at least 30 unique
   participants have high-confidence conflict predictions.

Statistical significance alone is insufficient.

If tau feasibility fails, stop the molecular-specificity claim without opening
tau values. If the AD reference does not beat the anatomical control, contract
the finding to generic classifier disagreement and do not call PARES-AD
AD-specific. If conditional or high-confidence gates fail, retain only the
unadjusted risk-enrichment result. Do not change biomarkers, regions, windows,
thresholds, covariates, or model subsets.

## 10. Permitted claims after this experiment

If all gates pass:

> PARES-AD provides a disease-specific, molecularly corroborated anatomical
> counterevidence layer that identifies high-risk and high-confidence-conflict
> AD MRI predictions across heterogeneous models and acquisition domains. It
> complements confidence-based ranking but does not replace it.

If only existing risk enrichment remains:

> PARES-AD provides an independently replicated anatomical-concordance warning,
> but its AD-biological specificity and incremental conditional value were not
> established.

Never claim internal causal mechanism, neuropathologic diagnosis, clinician
benefit, external-institution validity, or superiority to confidence ranking.

## 11. Runtime and artifacts

Use no new model training beyond low-dimensional reference/regression fits.
Expected runtime is 10–18 minutes on CPU, dominated by replay of 290 frozen
checkpoints. Stop if projected or elapsed time exceeds 30 minutes.

Generate only:

1. one aggregate result CSV;
2. one summary JSON;
3. one concise evidence report.

No participant identifier, participant-level result, anchor-value table,
checkpoint, model, intermediate tensor, figure, or manuscript package may be
written.
