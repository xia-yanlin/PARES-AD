# From manuscript to code

Paths are relative to `paper_scripts/research_v2/src/`. Scripts are run from the project root
(`python -m research_v2.src.<package>.<module>`, or `python <path>` for the dated P1/P2 scripts) and read the
restricted data from `data/` and `research_v2/outputs/private/`. Protocols referenced by a script are in `protocols/`.

| Manuscript element | Script(s) |
| --- | --- |
| Acquisition domains, nested splits, input preprocessing (Section 2.1–2.2, S1–S2) | `data/build_acquisition_domains.py`, `data/build_nested_domain_splits.py`, `data/noncombat_features.py`, `data/roi_mapping.py` |
| Nine frozen classifiers: inner-domain benchmark, eligibility, outer refits (Section 2.2, S2, D1, D14, D15) | `execution/build_fw2_job_manifest.py`, `execution/run_fw2_job.py`, `execution/run_fw2_shard.py`, `execution/aggregate_fw2_inner.py`, `execution/build_fw5_manifest.py`, `execution/run_fw5_job.py`, `execution/run_fw5_to_completion.py`, `execution/aggregate_fw5_outer.py`, `models/neural.py` |
| Anatomical reference, auxiliary regression, residual rule, case reports (Figs. 1–4, S2–S3, D2) | `analysis/build_ad_mri_explanation_system.py` |
| ADNI primary endpoint, disagreement RR, designated control, adjusted OR (Table 2, Figs. 5–6, S3–S4, D3) | `analysis/confirm_pares_ad_independent.py` |
| Five-group report analysis (Table S3) | `analysis/confirm_pares_ad_state_confirmation.py` |
| Clinical and molecular endpoints with the scale correction (Table S4) | `analysis/validate_pares_ad_conditional_specificity.py`, `analysis/run_pares_ad_anchor_validation.py`, `analysis/audit_pares_ftp_feasibility.py`, `analysis/build_pares_anchor_receipt.R`, `analysis/audit_anchor_scale_correction_2026_10_09.py` |
| Candidate references (Section 3.3, S4, D4, D16) | `analysis/run_pares_performance_matched_reference.py`, `analysis/repair_pares_fixed_candidate_evaluation.py` |
| NACC input bridge, linkage, frozen external evaluation (Table 3, Fig. 7, S5, D5) | `analysis/audit_nacc_source_input_bridge.py`, `analysis/freeze_nacc_external_predictions.py`, `analysis/evaluate_nacc_external.py`, `analysis/diagnose_nacc_qc_linkage.py`, `analysis/diagnose_nacc_control_risk_posthoc.py` |
| NACC review intervals and CDR-SB (S6–S7, Table S8, D6) | `analysis/strengthen_nacc_manuscript.py`, `analysis/verify_nacc_manuscript_strengthening.py` |
| ADNI error-ranking AUROC increment (Section 3.6, S6) | `analysis/evaluate_pares_ad_error_risk_increment.py` |
| Four review scores, error types and case composition (Table 4, Figs. 8–9, S6–S7, D7–D12) | `analysis/audit_pares_ad_comparators.py`, `analysis/verify_pares_ad_comparators.py` |
| Repeat scans (S8, D13) | `analysis/check_pares_observed_pair_feasibility.py`, `analysis/evaluate_pares_observed_pairs.py`, `analysis/verify_pares_observed_pairs.py` |
| Reconstruction variants (Table S9, D17–D18) | `analysis/develop_pares_additive_behavior.py`, `analysis/develop_pares_teacher_consistency.py`, `analysis/evaluate_pares_teacher_consistency_pairs.py`, `analysis/check_pares_source_pair_feasibility.py`, `analysis/develop_pares_direct_change.py`, and the corresponding `analysis/verify_*.py` |
| Case-mix reweighting, control and candidate comparisons (S9, D19–D21) | `analysis/pares_ad_p1_supplement_2026-10-14.py` |
| Committee and locally refit comparators (Table 5, S10, D22–D23) | `analysis/pares_ad_p2_ensemble_baseline_2026-10-14.py`; ADNI local refit with protocol weights: `corrections/pares_ad_p2_adni_weight_correction_2026-10-15.py` |
| Aggregate tables and figure data | `reporting/build_pares_ad_manuscript_assets.py` |

Figure layout scripts and the raw-data download steps are not included.

## Checks

1. `python tools/run_tests.py` — synthetic unit tests of the library.
2. `python tests/parity_with_saved_scores.py --root <project root>` — restricted data. Refits the source-fitted
   heads C, A, K and P in ADNI and both NACC cohorts and compares them with the saved scores; reruns the ADNI
   fixed-budget bootstrap (2,000 participant-cluster resamples) and compares A−C and P−C with D8; recomputes the
   committee and locally refit scores in all three cohorts and compares their yields with Table 5 / D22.
   The result of the run on the original data is in [`validation/PARITY_RESULT.txt`](validation/PARITY_RESULT.txt).
3. Research test suite: from `paper_scripts/`, `python -m pytest research_v2/tests` (requires the environment in
   `paper_scripts/requirements.txt` and pytest).
