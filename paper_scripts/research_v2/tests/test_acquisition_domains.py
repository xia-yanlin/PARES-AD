from __future__ import annotations

import unittest

import pandas as pd

from research_v2.src.data.build_acquisition_domains import (
    PHASE_MANUFACTURER_ORDER,
    build_assignments,
    canonicalize_field_strength,
    canonicalize_manufacturer,
    is_primary_eligible,
    is_stable_eligible,
)


class AcquisitionDomainTests(unittest.TestCase):
    def test_manufacturer_variants_are_canonicalized(self) -> None:
        self.assertEqual(canonicalize_manufacturer("SIEMENS"), "Siemens")
        self.assertEqual(canonicalize_manufacturer("Siemens Healthineers"), "Siemens")
        self.assertEqual(canonicalize_manufacturer("GE MEDICAL SYSTEMS"), "GE")
        self.assertEqual(canonicalize_manufacturer("Philips Healthcare"), "Philips")

    def test_unknown_manufacturer_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            canonicalize_manufacturer("Unknown Vendor")

    def test_field_strength_is_canonicalized(self) -> None:
        self.assertEqual(canonicalize_field_strength(1.5), "1.5T")
        self.assertEqual(canonicalize_field_strength("3.0"), "3.0T")
        with self.assertRaises(ValueError):
            canonicalize_field_strength(7.0)

    def test_eligibility_thresholds_are_inclusive(self) -> None:
        self.assertTrue(is_primary_eligible(5, 10))
        self.assertFalse(is_primary_eligible(4, 100))
        self.assertTrue(is_stable_eligible(10, 20))
        self.assertFalse(is_stable_eligible(9, 100))

    def test_each_subject_receives_exactly_one_domain(self) -> None:
        rows = []
        rid = 1
        manufacturer_raw = {
            "Siemens": "Siemens Healthineers",
            "GE": "GE MEDICAL SYSTEMS",
            "Philips": "Philips Healthcare",
        }
        for domain in PHASE_MANUFACTURER_ORDER:
            phase, manufacturer = domain.split("_", maxsplit=1)
            rows.append(
                {
                    "RID": rid,
                    "PHASE": phase,
                    "ScannerManufacturer": manufacturer_raw[manufacturer],
                    "FieldSgth": 1.5 if phase == "ADNI1" else 3.0,
                    "AD_label": rid % 2,
                }
            )
            rid += 1
        assignments = build_assignments(pd.DataFrame(rows))
        self.assertEqual(len(assignments), len(rows))
        self.assertFalse(assignments["acquisition_domain"].isna().any())
        self.assertEqual(set(assignments["acquisition_domain"]), set(PHASE_MANUFACTURER_ORDER))


if __name__ == "__main__":
    unittest.main()
