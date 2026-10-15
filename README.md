# PARES-AD

An anatomical reference framework for MRI classifier error review in Alzheimer's disease.

PARES-AD compares an MRI classifier's predictions with a transparent, four-feature anatomical reference: hippocampal volume, entorhinal thickness, ventricular volume, and temporoparietal thickness. It provides itemized reference contributions and scores for prioritizing predictions for review under a fixed workload.

This repository contains the Python library, synthetic examples, tests, and analysis scripts accompanying the PARES-AD manuscript. Current code version: **1.1.0**.

## Installation

Requires Python 3.10 or newer. The library depends on NumPy, pandas, and SciPy.

```sh
git clone https://github.com/xia-yanlin/PARES-AD.git
cd PARES-AD
python -m pip install .
```

For the synthetic tests and examples:

```sh
python -m pip install -r requirements-test.txt
python tools/run_tests.py
python demo/run_demo.py 200
```

The example generates synthetic source and target cohorts and reports fixed-budget review yields with 200 bootstrap resamples. It does not require ADNI or NACC data.

## Repository structure

| Directory | Contents |
| --- | --- |
| [`pares_ad/`](pares_ad/) | Anatomical reference, case reports, review scores, local cross-fitting, and bootstrap evaluation |
| [`demo/`](demo/) | Synthetic data and a runnable example |
| [`tests/`](tests/) | Synthetic tests and restricted-data parity checks |
| [`paper_scripts/`](paper_scripts/) | Original paper analysis scripts and the separate ADNI weighting correction |
| [`protocols/`](protocols/) | Dated analysis protocols |
| [`docs/`](docs/) | Input requirements, reproduction guide, and implementation notes |

## Using the library

The library includes confidence-only scores (`C`), binary anatomical disagreement (`A`), continuous reference opposition (`P`), and a committee comparator (`CE`). Locally fitted comparators use participant-level cross-fitting.

See the [runnable example](demo/run_demo.py) and [data contract](docs/DATA_CONTRACT.md) for input columns and usage.

## Reproducing the paper

The [reproduction guide](docs/REPRODUCIBILITY.md) maps manuscript results to analysis scripts. [Implementation notes](docs/NOTES.md) describe weighting, review budgets, uncertainty estimates, and the ADNI local-refit correction. [Validation records](docs/validation/) document synthetic tests, saved-result comparisons, and source provenance.

No participant-level data, predictions, or model checkpoints are distributed. Reproducing the original analyses requires authorized access to [ADNI](https://ida.loni.usc.edu/) and [NACC](https://naccdata.org/) data and the project inputs described in the reproduction guide. The synthetic example runs independently of those data.

## Citation and license

If you use this code, please cite *PARES-AD: An anatomical reference framework for MRI review in Alzheimer's disease*. Publication details will be added when available.

[MIT License](LICENSE).
