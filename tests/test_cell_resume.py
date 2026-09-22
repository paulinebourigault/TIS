"""An interrupted LLM study must resume from cached cells, byte-identically."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tis.llm_study import runner  # noqa: E402
from tis.llm_study.mock import write_mock_kernel_artifact  # noqa: E402


class CellResumeTests(unittest.TestCase):
    def test_second_run_uses_cache_only_and_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            kernels = root / "mock.json"
            write_mock_kernel_artifact(kernels, questions=2)
            config = {
                "schema_version": 1,
                "master_seed": 5,
                "replications": 2,
                "horizons": [2],
                "alphas": [0.1],
                "budgets_per_shared_group": [50],
                "methods": ["uniform", "tis", "untied_tis"],
                "pilot_multiplier": 4.0,
                "minimum_pilot_per_group": 8,
                "minimum_main_per_group": 2,
                "exploration_exponent": 0.25,
                "output_directory": "unused",
            }
            config_path = root / "configs" / "config.json"
            config_path.write_text(json.dumps(config))
            first = runner.run_llm_study(
                config_path, kernels, output_directory=root / "first"
            )
            cache_root = root / "build" / "llm_cell_cache"
            shards = list(cache_root.rglob("*.json.gz"))
            self.assertEqual(len(shards), 2)
            with mock.patch.object(
                runner,
                "_question_cell",
                side_effect=AssertionError("cache miss: cell recomputed"),
            ):
                second = runner.run_llm_study(
                    config_path, kernels, output_directory=root / "second"
                )
            for name in (
                "raw.csv", "question_summary.csv", "panel_replications.csv",
                "panel_summary.csv", "paired_panel.csv",
                "allocation_summary.csv", "diagnostics.json",
            ):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    f"{name} differs between fresh and cache-resumed runs",
                )
            second_manifest = json.loads((second / "manifest.json").read_text())
            self.assertEqual(second_manifest["cells_from_cache"], 2)

    def test_kernel_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            first_kernels = root / "mock1.json"
            second_kernels = root / "mock2.json"
            write_mock_kernel_artifact(first_kernels, questions=1)
            write_mock_kernel_artifact(second_kernels, questions=2)
            config = {
                "schema_version": 1,
                "master_seed": 5,
                "replications": 1,
                "horizons": [2],
                "alphas": [0.1],
                "budgets_per_shared_group": [50],
                "methods": ["uniform", "tis"],
                "pilot_multiplier": 4.0,
                "minimum_pilot_per_group": 8,
                "minimum_main_per_group": 2,
                "exploration_exponent": 0.25,
                "output_directory": "unused",
            }
            config_path = root / "configs" / "config.json"
            config_path.write_text(json.dumps(config))
            runner.run_llm_study(
                config_path, first_kernels, output_directory=root / "a"
            )
            runner.run_llm_study(
                config_path, second_kernels, output_directory=root / "b"
            )
            signatures = {
                path.name for path in (root / "build" / "llm_cell_cache").iterdir()
            }
            self.assertEqual(len(signatures), 2)


if __name__ == "__main__":
    unittest.main()
