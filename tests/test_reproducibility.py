from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

from tis.artifact import verify_result_directory
from tis.runner import run_configuration
from tis.runner import METHODS


class ReproducibilityTests(unittest.TestCase):
    def test_csv_outputs_are_byte_identical(self) -> None:
        repository = Path.cwd()
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            root = Path(temporary)
            config_directory = root / "configs"
            config_directory.mkdir()
            common = {
                "schema_version": 1,
                "master_seed": 77,
                "replications": 2,
                "environments": [
                    {
                        "name": "controlled_shared",
                        "alpha": 0.1,
                        "budgets_per_group": [20],
                    }
                ],
                "pilot_multiplier": 1.0,
                "minimum_pilot_per_group": 2,
                "minimum_main_per_group": 2,
                "exploration_exponent": 0.25,
            }
            paths = []
            for index in (1, 2):
                config = dict(common)
                config["output_directory"] = str(root / f"result-{index}")
                path = config_directory / f"run-{index}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                paths.append(run_configuration(path))
            for filename in (
                "raw.csv",
                "summary.csv",
                "paired.csv",
                "allocations.csv",
                "table.tex",
                "diagnostics.json",
            ):
                self.assertEqual(
                    (paths[0] / filename).read_bytes(),
                    (paths[1] / filename).read_bytes(),
                    filename,
                )
            self.assertTrue(verify_result_directory(paths[0])["passed"])

            with (paths[0] / "raw.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["method"] for row in rows}, set(METHODS))
            by_method = {row["method"]: row for row in rows}
            learned = ("mc_ucb_frozen", "tis", "tail_no_cov", "learned_mean")
            self.assertEqual(
                {int(by_method[method]["pilot_total"]) for method in learned},
                {int(by_method["tis"]["pilot_total"])},
            )
            self.assertEqual(
                {int(by_method[method]["main_total"]) for method in learned},
                {int(by_method["tis"]["main_total"])},
            )
            self.assertTrue(
                all(int(by_method[method]["pilot_total"]) == 0 for method in
                    ("uniform", "occupancy", "mean_influence", "oracle_tail", "oracle_no_cov"))
            )


if __name__ == "__main__":
    unittest.main()
