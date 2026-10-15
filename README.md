# PARES-AD

An anatomical reference framework for MRI classifier error review in Alzheimer's disease.

This private repository stores the **1.1.0 submission code package**, byte-for-byte identical to Supplementary Material ESM_3 prepared on 15 October 2026. The complete source tree is inside [ESM_3.zip](ESM_3.zip), including the reusable `pares_ad` library, synthetic examples and tests, 90 original research files, the ADNI weighting correction script, and 19 dated protocols.

## Download and run

Download `ESM_3.zip` and extract it. From the extracted `pares_ad_code` directory:

```sh
python tools/verify_manifest.py
python -m pip install .
python -m pip install -r requirements-test.txt
python tools/run_tests.py
python demo/run_demo.py 200
```

Python 3.10 or newer is required. The package README and `docs/REPRODUCIBILITY.md` give the full usage instructions and map manuscript results to analysis scripts. The manifest verifies a clean extracted package before installation creates local build files.

## Version and verification

- Version: 1.1.0.
- 17 synthetic tests passed; offline local installation and installed-package import verified.
- The package includes the original restricted-data parity check and its run record for ADNI and both NACC cohorts. This is a check of the stated saved-score and aggregate comparisons, not a complete rerun from raw MRI data.
- The corrected ADNI local-refit block uses the configuration-equal training weights specified in the protocol. The original analysis and the separate correction are retained.
- Archive SHA-256: `487f491836529e55caa0753e0eb0f0b19e88a026156e00e52888f7602f3d061f`.

No participant-level data, predictions, model checkpoints or access credentials are included. Original-data reproduction requires authorized ADNI/NACC access and the project inputs described inside the archive. The package contains its MIT license.

The repository is private pending publication. No public release has been made.
