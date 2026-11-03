"""Regression and failure tests for analysis-only verification (no simulator).

Run with unittest so the legacy simulator fixtures in conftest.py aren't loaded.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_results as verify


class PublishedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = verify.read_exports(ROOT)

    def test_published_results_pass(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify.verify_results(ROOT)

    def test_changed_summary_value_fails(self):
        frames = copy.deepcopy(self.frames)
        frames[f"{verify.PREFIX}_no_occupant_mean_building_kpis_february.csv"].loc[0, "Ramping"] += 0.01
        with self.assertRaisesRegex(verify.VerificationError, "numerical values differ"):
            verify.verify_tables(frames)

    def test_changed_building_metric_fails(self):
        frames = copy.deepcopy(self.frames)
        frames[f"{verify.PREFIX}_building_kpis_augmented.csv"].loc[0, "kelvin_hours"] += 1
        with self.assertRaisesRegex(verify.VerificationError, "numerical values differ"):
            verify.verify_tables(frames)

    def test_missing_building_fails(self):
        buildings = self.frames[f"{verify.PREFIX}_building_kpis_augmented.csv"].iloc[1:]
        with self.assertRaisesRegex(verify.VerificationError, "coverage differs"):
            verify.verify_buildings(buildings)

    def test_duplicate_building_fails(self):
        buildings = self.frames[f"{verify.PREFIX}_building_kpis_augmented.csv"].copy()
        buildings.iloc[1] = buildings.iloc[0]
        with self.assertRaisesRegex(verify.VerificationError, "duplicate"):
            verify.verify_buildings(buildings)

    def test_wrong_run_fails(self):
        buildings = self.frames[f"{verify.PREFIX}_building_kpis_augmented.csv"].copy()
        buildings.loc[0, "case"] = "another_experiment"
        with self.assertRaisesRegex(verify.VerificationError, "experiment identifier"):
            verify.verify_buildings(buildings)

    def test_changed_override_delta_fails(self):
        frames = copy.deepcopy(self.frames)
        frames[f"{verify.RUN}_override_deltas.csv"].loc[0, "delta_overrides"] += 1
        with self.assertRaisesRegex(verify.VerificationError, "numerical values differ"):
            verify.verify_overrides(frames)

    def test_reordered_summary_rows_pass(self):
        frames = copy.deepcopy(self.frames)
        name = f"{verify.PREFIX}_no_occupant_mean_building_kpis_february.csv"
        frames[name] = frames[name].iloc[::-1]
        verify.verify_tables(frames)

    def test_artifact_corruption_and_manifest_omission_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis"
            analysis.mkdir()
            manifest_path = analysis / "artifact_manifest.json"
            shutil.copy(ROOT / "analysis/artifact_manifest.json", manifest_path)
            manifest = json.loads(manifest_path.read_text())
            for item in manifest["artifacts"]:
                target = root / item["published"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / item["published"], target)
            artifact = root / manifest["artifacts"][0]["published"]
            with artifact.open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaisesRegex(verify.VerificationError, "SHA-256 mismatch"):
                verify.verify_manifest(root)
            manifest["artifacts"].pop(0)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(verify.VerificationError, "Manifest does not cover"):
                verify.verify_manifest(root)

    def test_missing_nonfinite_and_nonnumeric_kpis_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "analysis/data", root / "analysis/data")
            name = f"{verify.PREFIX}_building_kpis_augmented.csv"
            for value in [np.nan, np.inf, "invalid"]:
                with self.subTest(value=value):
                    frame = self.frames[name].copy()
                    frame["kelvin_hours"] = frame["kelvin_hours"].astype(object)
                    frame.loc[0, "kelvin_hours"] = value
                    frame.to_csv(root / "analysis/data" / name, index=False)
                    with self.assertRaises(ValueError):
                        verify.read_exports(root)


if __name__ == "__main__":
    unittest.main()
