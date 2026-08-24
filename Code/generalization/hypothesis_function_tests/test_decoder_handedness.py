"""Tests for animal-specific reaching-arm marker selection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Code.decoder_utils import parse_peak_time, reach_marker_names, reach_side_for_session


class TestDecoderHandedness(unittest.TestCase):
    def test_tria_uses_right_arm(self):
        session = "TSAL20250813_0830_staticAndStaticFree001"

        self.assertEqual(reach_side_for_session(session), "r")
        self.assertEqual(reach_marker_names(session), ("r-wrist", "r-shoulder"))

    def test_tony_uses_left_arm(self):
        session = "TYTR20250304_0830_interferenceAndInterferenceFree001"

        self.assertEqual(reach_side_for_session(session), "l")
        self.assertEqual(reach_marker_names(session), ("l-wrist", "l-shoulder"))

    def test_unknown_animal_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "Unknown reaching side"):
            reach_marker_names("ZZXX20260101_unknown")

    def test_empty_peak_time_is_missing(self):
        row = pd.Series({"peak_extension_times": ""})
        self.assertTrue(np.isnan(parse_peak_time(row)))

    def test_peak_time_uses_last_sequence_value(self):
        row = pd.Series({"peak_extension_times": [1.0, 2.5]})
        self.assertEqual(parse_peak_time(row), 2.5)


if __name__ == "__main__":
    unittest.main()
