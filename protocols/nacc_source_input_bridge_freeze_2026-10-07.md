# NACC missing-input bridge: source-only technical assessment

Frozen before masked source replay and before any NACC model inference.

## Purpose

NACC75 lacks bilateral bankssts, frontalpole and temporalpole thickness and volume (12 of 163 required numeric inputs including eTIV). All four original AD-reference concepts and the equal-complexity control inputs are available. Establish whether the existing saved training-median missing-value handler materially changes source behavior under this exact structural missingness pattern. This is a source-only measurement/implementation assessment for the authorized PARES-AD external extension, not a revival of concept-use model selection or a replacement external endpoint.

## Frozen computation

- Use only the original ADNI development outer predictions and their 650 participants. Do not use the 882-person confirmation cohort for this diagnostic.
- Replay all 290 existing checkpoint jobs, existing preprocessors, graphs and Platt calibrators, aggregating the five seeds for each available configuration/domain exactly as before.
- First reproduce full-input saved development probabilities (maximum absolute difference <= 1e-6). Stop on checkpoint-integrity or replay mismatch.
- Then set exactly the 12 unavailable columns to NaN and use each checkpoint's previously fitted training-local median imputation. No NACC fitting, no nearest-ROI substitution, no retraining, no parameter/threshold search and no favorable configuration selection.
- Report probability absolute differences, class disagreement, and AUROC/Brier changes overall and for every configuration and family. Missing or single-class metrics must remain non-estimable.
- Prespecified materiality flags: absolute AUROC loss > 0.01 or Brier increase > 0.01 overall or in any configuration. These are engineering decision thresholds, not inferential non-inferiority claims. Report all results irrespective of direction.
- If any flag occurs, do not treat missing-column transport as negligible. Pursue the missing measurements or a separately described shared-input model development design; do not quietly label a changed classifier as original-model validation.
- Even if no flag occurs, an external evaluation using this input policy must be explicitly described as a missing-input deployment condition. Source agreement cannot establish NACC transport, external calibration, or participant independence.

Acquisition phase handling remains unresolved separately: NACC must not be relabeled ADNI1/3/4. The full five-state external contract cannot be claimed from this input diagnostic alone.
