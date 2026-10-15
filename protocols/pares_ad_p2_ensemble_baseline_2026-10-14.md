# PARES-AD P2 supplement: cross-classifier disagreement baseline (post hoc, written before the run)

Date written: 2026-10-14, before any P2 result was computed.
Status: post hoc / exploratory, authorized by the corresponding author on 2026-10-14. No MRI training or inference; saved prediction scores only. Aggregate outputs only.

## Purpose
Reviewers of selective-prediction and error-detection work expect a standard reference-free baseline besides softmax confidence. The nine frozen classifiers already give several predictions per participant, so the committee-style disagreement of those predictions is available without retraining.

## Definitions (fixed before running)
For participant i with prediction rows r in R_i (all configurations, and all source folds in NACC), p_ir = classifier probability of AD:
- E_sd(i) = population standard deviation of p_ir over R_i.
- f(i) = share of rows with p_ir >= 0.5; E_vote(i) = 4 f (1 - f).
Review scores:
- CE (primary baseline): StandardScaler + logistic (C = 0.1) on [|z|, |z|^2, predicted class, E_sd, E_vote], z = logit of the row probability; same source data, weights, and target (classifier error) as the C score. Fit once on source predictions (ADNI: 650-person source predictions; NACC: masked replay).
- CEs (sensitivity): same with E_sd only.
Contrasts: CE minus C, A minus CE, P minus CE at 5%, 10%, 20% budgets, same budget rule and bootstrap (participant 2,000; center 1,000 for NACC) as the main fixed-budget analysis. Saved C, A, P scores are used unchanged.

## Interpretation rule (fixed before running)
Report all three cohorts and all budgets regardless of direction. A and P are described as better than the committee baseline only where the participant-cluster interval excludes zero.

## Addendum A (written after seeing the first CE and CEs results, before running CEr)
The first run showed that the committee spread E_sd is about three times larger in both NACC cohorts (mean 0.09-0.10) than in the source predictions (0.032), so heads fit on source predictions transport poorly (CE minus C was negative in NACC). To give the committee baseline its best chance, one further variant was added:
- CEr: same as CE, but E_sd and E_vote are replaced by their participant-level percentile ranks within the cohort being scored (source set for fitting; the target cohort for scoring). This uses no labels and removes the scale shift.
CE, CEs, and CEr are all reported. CEr is post hoc relative to the original two variants.

## Addendum B (written after seeing the CEr results, before running the locally refit sensitivity)
CE, CEs, and CEr all fell below the confidence score C in both NACC cohorts, even though they contain the same confidence features. This can mean either that committee spread is uninformative about errors there, or that heads fit on source predictions do not transport. To tell these apart, one label-using sensitivity analysis was added (upper bound for every score, not a deployable procedure):
- Cross-fitted local refit: within each cohort, participants are split into five folds (seed 20261014); for each fold, the heads for C, CE, A, and P are refit on the other four folds (config-equal weights from review_weight, same features, same logistic settings, target = classifier error) and scored on the held-out fold. The saved C/A/P scores are not used in this analysis. Contrasts: CE minus C, P minus C, P minus CE, A minus CE (labels Cx, CEx, Ax, Px).
- Bootstrap treats the cross-fitted scores as fixed, as in the main analysis (heads fit once), so intervals ignore refit variability.
This analysis is post hoc and is reported as a sensitivity analysis.
