# PARES-AD P1 supplement: prevalence reweighting and flag-matched controls

Date: 2026-10-14. The user explicitly authorized these two computational analyses
in this session ("补上"). This protocol is recorded BEFORE the analyses are run.
All results are post hoc / exploratory: ADNI and NACC outcomes and all prior
analyses were already known. Nothing here is a new confirmation, and none of it
replaces a frozen endpoint (the ADNI primary endpoint stays negative; the frozen
NACC AD-minus-control RR difference stays as reported).

## Scope (what is NOT done)
No MRI training or inference, no new ROI or threshold search beyond the single
pre-stated flag-rate matching rule below, no new disease/anchor, no manuscript
edit. Fixed saved predictions and the existing source-fitted risk heads are
reused. Historical result directories are read-only; outputs go to
`research_v2/outputs/private/pares_ad_p1_supplement_2026-10-14/` (aggregate
tables only; no participant-level rows are written).

## Stage 0: reproduction gate
Before any new number is interpreted, the script must (a) refit C/A/K/P heads on
the saved source predictions and reproduce the saved cohort scores (max abs
difference < 1e-5), and (b) reproduce D8 A-minus-C yield differences in ADNI,
NACC sequence-ID and NACC date/description at 5/10/20% (point estimates to
0.01 pp). If the gate fails, nothing downstream is reported.

## P1-a: case-mix (prevalence) transport of the A-minus-C review gain
Estimand: review-yield difference (A minus C) at the 5/10/20% budgets after
reweighting participants so that the AD share equals a target pi, with the saved
fixed scores (heads are NOT refitted). Reweighting is at participant level
(AD and CN participants get constant factors, recomputed in every bootstrap
draw); the review budget is a fraction of the reweighted workload (fractional
last row). Participant-cluster bootstrap, 2,000 draws; NACC additionally
center-cluster (1,000 draws).
Scenarios: ADNI reweighted to the NACC date/description AD share (491/3,797)
and to the NACC sequence-ID share (32/145); NACC date/description and sequence-ID
reweighted to the ADNI share (342/882). An identity check reweights ADNI to its
own share (should reproduce native values within rounding of the budget rule).
Pre-stated reading (10% budget, A minus C):
- Case-mix account SUPPORTED if ADNI at the NACC date/description share has a
  point estimate >= +5.0 pp with 95% lower bound > 0, AND NACC date/description
  at the ADNI share has a point estimate < +5.0 pp or a lower bound <= 0.
- PARTIAL if only one of the two holds, or the ADNI point estimate is +2 to +5 pp.
- NOT SUPPORTED if ADNI at the low share stays < +2 pp (the NACC gain would then
  need another explanation: cohort, protocol, missing-input bridging or linkage).
Both directions and all budgets are reported regardless of outcome.

## P1-b: flag-rate-matched control and performance-comparable candidates
Motivation: in NACC the specified control predicts AD for 0.3% of predictions,
so its binary "disagreement" is almost exactly "the classifier predicts AD".
Rule 1 (control). Choose the control decision threshold t on SOURCE predictions
only (config-equal weights, same source set used to fit each head) so that the
control's binary-disagreement flag rate equals the AD reference's flag rate at
0.5; grid t in {0.01, 0.015, ..., 0.99}, choose the minimum absolute rate gap
(first on ties). Fit K' = C + that disagreement, with the unchanged head recipe
(StandardScaler + logistic regression, C = 0.1, config-equal weights). ADNI uses
the full-input source predictions, NACC the masked-replay source predictions.
Rule 2 (candidates, ADNI only; NACC candidate probabilities do not exist). For
each of the 10 fixed candidate references: K_c at the native 0.5 threshold and
K_c at a source-matched flag rate by Rule 1.
Reported: C, A, K, K', K_c yields, and contrasts A-C, K'-C, A-K', A-K_c,
K_c-C at 5/10/20% with participant-cluster bootstrap intervals (NACC also
center-cluster); weighted RR point estimates of each flagger.
Pre-stated reading (10% budget): the AD reference "retains an advantage over a
flag-rate-matched non-AD flagger" in a cohort if A-K' has a lower bound > 0;
for the candidates, the count of candidates with A-K_c lower bound > 0 is
reported at each budget. If K' or K_c matches A, the correct reading is that an
independent second reader supplies the review gain and anatomy-specificity is
not demonstrated; this is reported as such.

## Provenance
Script, protocol and outputs are hashed in the output receipt. Seeds: participant
bootstrap native replication 20261018 and center 20261019 (identical to the
original stream so D8 intervals can be matched); new scenarios 20261014+k.
