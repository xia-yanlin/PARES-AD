# PARES-AD content audit: bounded numeric correction

2026-10-09. Authorized by the user's request to finish all possible content
checks, excluding English and figure/table production. This is an error audit
of already reported endpoints, not a new experiment or outcome search.

Inspection found that `crossfit_continuous` returns a centered/scaled residual
but unscaled residual predictions to `continuous_bootstrap`. The point effect
uses the untransformed residual. Thus point and interval may use different
estimators. Reproduce the exact historical PACC and tau points/intervals first;
then compute intervals using the original residual and matching predictions.
Keep original participants, links, covariates, fold order/seed, score fits,
2,000 bootstrap draws and seed unchanged. Retain standardized residuals ONLY
for the separately defined signed adjudication endpoint. No new MRI inference,
ROI, thresholds, samples, refit-bootstrap design or outcome is permitted.

Use saved model predictions and fixed-source reference recipes, validate source
hashes and historical reproduction. Preserve all historical outputs. Publish
aggregate correction results in a new dated directory and independently check
the corrected SSE formulas with explicit resampled indices. If the original
receipt cannot be reproduced, do not silently replace it. Continuous-anchor
effect is residual-SSE-relative improvement, not conventional delta R² with
total outcome variance as denominator; make that distinction from NACC CDR-SB.

The current manuscript tables/figures are not to be regenerated. Any affected
legacy numbers in deferred assets must be explicitly listed as awaiting sync;
the Chinese scientific text must identify the authoritative corrected values.
