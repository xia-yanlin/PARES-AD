# PARES-AD evidence-state confirmation freeze

**Date:** 2026-09-27  
**Purpose:** confirm whether the complete PARES-AD evidence-state contract, not only the pathology-conflict flag, can be transported from the development system to the 882-participant development-isolated ADNI cohort.  
**Status:** frozen before reading the new confirmation-state/error results.

## 1. Question

When the development-fitted PARES-AD reference, behavior-reconstruction parameters, residual bound, and acquisition-validation gate are applied without refitting to the development-isolated ADNI cohort, do the five prespecified evidence states retain distinct error-risk profiles?

This is a confirmation of an audit output contract. It is not a new classifier, a prospective model-selection experiment, an external-program validation, or a clinical-use study.

## 2. Data roles

- Development role: the existing 650 participants, 5,184 model–case predictions, 9 configurations, 3 families and 7 held-out acquisition domains. These data fit the transparent reference, behavior regressions, nested residual calibration and acquisition-validation gate.
- Confirmation role: the existing 882 participants and 7,091 frozen model–case predictions. Their diagnosis labels are inaccessible to state assignment and are used only after state assignment for error-risk evaluation.
- No classifier checkpoint, calibration parameter, model family or confirmation label is refit or used to define a state.

## 3. Frozen state assignment

For each configuration and outer acquisition domain, apply the development-fitted two-stage reconstruction to the confirmation cases in that domain. Use the development nested-training 90% residual bound. Apply the development configuration-level acquisition-validation gate without recomputation on confirmation data.

State precedence is fixed:

1. residual outside the bound → `UNEXPLAINED_ABSTAIN`;
2. acquisition-dominant component and development acquisition gate invalid → `ACQUISITION_UNVALIDATED_ABSTAIN`;
3. acquisition-dominant component and development acquisition gate valid → `ACQUISITION_SENSITIVE`;
4. model/reference threshold conflict → `PATHOLOGY_CONFLICT`;
5. otherwise → `PATHOLOGY_ALIGNED`.

State assignment uses model probability, transparent reference probability, four structural contributions, acquisition metadata, frozen reconstruction parameters and residual bound only. It does not use diagnosis, error, PACC, tau, amyloid or any confirmation outcome.

## 4. Prespecified summaries

Report, without post-hoc merging or deletion:

- state counts, fractions and acquisition-domain coverage;
- state-specific error rates and participant-cluster bootstrap intervals;
- risk ratio for each non-aligned state versus `PATHOLOGY_ALIGNED`;
- direction across configurations, families and acquisition domains where estimable;
- state assignments with no outcome labels in the output artifact.

The primary confirmation state is `PATHOLOGY_CONFLICT` versus `PATHOLOGY_ALIGNED`, because this is the only state with an existing independent result and a frozen external confirmation claim. `UNEXPLAINED_ABSTAIN` is a prespecified secondary reliability state. The two acquisition states are descriptive and remain unvalidated if their development gate is not passed; they cannot be upgraded by a favorable error rate.

## 5. Interpretation gates

- If pathology conflict retains an error-risk interval above one and the direction is preserved across the frozen configuration/family structure, the existing main result is strengthened from a binary warning to a reusable state-contract component.
- If unexplained abstention retains a positive risk association, the abstention output gains empirical reliability meaning; otherwise it remains a coverage boundary only.
- Acquisition states are not interpreted as mechanisms unless their frozen acquisition-validity gate is satisfied; a null or mixed result is a valid stopping result.
- Failure of the full state contract does not erase the already confirmed pathology-conflict result and does not justify selecting a nicer subset of states.

## 6. Claim boundary

Success supports an ADNI-internal, development-isolated confirmation of an evidence-state audit contract. It does not support external transport, clinician/patient benefit, molecular specificity, causal internal mechanisms or an audit-guided model selector.
