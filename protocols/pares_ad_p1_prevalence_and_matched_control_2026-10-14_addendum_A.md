# Addendum A to the P1 protocol (2026-10-14)

Provenance note. This addendum was written in the working copy of the protocol at about 08:49 UTC on 2026-10-14, after a 40-resample smoke run and before the full-run stages (outputs written 08:52-08:56 UTC). The protocol file committed to this folder at that time was the version without the addendum; the omission was noticed and corrected at about 09:50 UTC the same day. The original protocol file is unchanged.

## Addendum A (2026-10-14, after a B=40 smoke run, before the full run)
Disclosure: a 40-draw smoke run of the first script version was executed to test
the code. I saw its P1-a point estimates (ADNI reweighted to the date/description
share: A-minus-C about +0.8 pp at 10%; date/description reweighted to the ADNI
share: about +4.8 pp). The P1-a rules above were NOT changed after seeing them.
Two protocol items could not be executed as written and are changed here:
1. Gate tolerance. Refit heads differ from the saved sklearn-lbfgs scores by up to
   2.2e-3 in probability (the saved scores come from an early-stopped lbfgs, my
   Newton solver converges to the exact optimum). The 1e-5 requirement is replaced
   by: saved scores remain the authoritative fixed scores for P1-a, and the native
   D8 A-minus-C yield differences must reproduce to 0.01 pp (they do). P1-b uses
   Newton refits for ALL heads (C, A, K, P and the new ones) so that its
   contrasts are internally consistent; An-minus-Cn is reported next to the saved
   A-minus-C to show the effect of the solver.
2. Rule 1 (flag-rate matching) is infeasible by thresholding: the control's lowest
   achievable source flag rate (about 15.8%) exceeds the AD reference's (about 8.0%)
   at every threshold, because the control agrees with the classifier less often
   than the AD reference at any cut-off. The threshold giving the minimum flag-rate
   gap is reported as "Kmin" and explicitly labelled as NOT matched. A continuous
   control-opposition score Pk (same recipe as P with the control probability) and,
   for the 10 ADNI candidates, continuous Pc and binary Kc at 0.5 are added.
   Contrasts A-Kmin, P-Pk, A-Kc and P-Pc replace the matched-rate contrast. The
   pre-stated reading for the control is otherwise unchanged: if the continuous or
   minimum-flag control matches the AD reference, anatomy-specificity is not
   demonstrated.
