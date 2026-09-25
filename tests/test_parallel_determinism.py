"""Serial and multi-process LLM study runs must be byte-identical."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tis.llm_study.mock import write_mock_kernel_artifact  # noqa: E402
from tis.llm_study.runner import run_llm_study  # noqa: E402


class ParallelDeterminismTests(unittest.TestCase):
    def test_serial_and_parallel_outputs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernels = root / "mock.json"
            write_mock_kernel_artifact(kernels, questions=2)
            config = {
                "schema_version": 1,
                "master_seed": 11,
                "replications": 2,
                "horizons": [2, 4],
                "alphas": [0.1],
                "budgets_per_shared_group": [50],
                "methods": [
                    "uniform", "occupancy", "mean_influence", "answer_entropy",
                    "oracle_tail", "tis", "mc_ucb_frozen", "adaptive_neyman",
                    "untied_tis",
                ],
                "pilot_multiplier": 4.0,
                "minimum_pilot_per_group": 8,
                "minimum_main_per_group": 2,
                "exploration_exponent": 0.25,
                "output_directory": "unused",
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            serial = run_llm_study(
                config_path, kernels, output_directory=root / "serial", processes=1
            )
            parallel = run_llm_study(
                config_path, kernels, output_directory=root / "parallel", processes=2
            )
            compared = 0
            for name in (
                "raw.csv", "question_summary.csv", "panel_replications.csv",
                "panel_summary.csv", "paired_panel.csv", "allocation_summary.csv",
                "diagnostics.json",
            ):
                self.assertEqual(
                    (serial / name).read_bytes(),
                    (parallel / name).read_bytes(),
                    f"{name} differs between serial and parallel runs",
                )
                compared += 1
            self.assertEqual(compared, 7)


if __name__ == "__main__":
    unittest.main()
