# Implementation notes

**Solvers.** The library fits review heads with a Newton solver (StandardScaler without weights, L2 logistic
regression with C = 0.1 and an unpenalized intercept). The original source-fitted heads were fitted with
scikit-learn L-BFGS; refitted probabilities agree with the saved scores to within 2.3e-3, and the fixed-budget results
reproduce to the reported precision ([parity record](validation/PARITY_RESULT.txt)). The reference uses L-BFGS-B with nonnegative bounds.

**Weights.** Source heads give equal total weight to each configuration. NACC predictions are weighted equally
across configurations and across source folds within configurations (`review_weight`). The locally refit
comparators use the same configuration-equal `review_weight` for training (`crossfit_scores` default).

**ADNI local refit, 2026-10-15.** The first run of the P2 analysis fitted the ADNI local-refit heads with equal row
weights, whereas the protocol (`protocols/pares_ad_p2_ensemble_baseline_2026-10-14.md`, Addendum B) specifies
configuration-equal weights; ADNI configurations contribute 605 to 882 predictions each. The ADNI block was rerun
with the protocol weights, leaving rows, folds, seeds, features and bootstrap draws unchanged
(`paper_scripts/research_v2/src/corrections/pares_ad_p2_adni_weight_correction_2026-10-15.py`; the script first
reproduces the original run). Table 5 and S10 report the corrected values. The two versions differ by at most
0.44 percentage points in any yield, difference or interval limit; Supplementary Data D22 retains both.

**Committee score.** The committee inputs are, per participant, the population standard deviation of the predicted
AD probability and the normalized vote disagreement 4f(1 − f), where f is the share of predictions of AD
(0 for unanimous, 1 for evenly split predictions). All available predictions of the participant contribute: up to
nine in ADNI and 58 configuration–source-fold predictions in NACC.

**Fixed budgets.** Within each configuration (ADNI) or configuration–source-fold unit (NACC), the
floor(budget × n) highest-scoring predictions are selected, with ties broken by row key. Case-mix reweighting uses
the fractional selection rule (`floor=False`).

**Intervals.** Bootstrap intervals resample participants (or NACC centers) and recompute budgets and rankings with
the fitted scores held fixed; they do not include refit variability.

**Naming.** Stage labels `fw2` and `fw5` in `paper_scripts/` denote the inner-domain benchmark and the outer-domain
refits of the classifiers. In the original scripts, `pathology_probability_AD` is the AD-reference probability and
`conflict` is binary disagreement; `build_concept_target` computes the four reference features.
