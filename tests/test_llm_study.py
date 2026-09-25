from __future__ import annotations

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from tis.engine import compute_influences, solve_probability_form
from tis.llm_study.artifact import verify_llm_result_directory
from tis.llm_study.calibrate import validate_kernel_artifact
from tis.llm_study.common import write_jsonl
from tis.llm_study.mock import write_mock_kernel_artifact
from tis.llm_study.models import shared_question_model, untied_question_model
from tis.llm_study.panel import freeze_panel
from tis.llm_study.protocol import brier_utility, exact_return_grid
from tis.llm_study.runner import run_llm_study


class LLMProtocolTests(unittest.TestCase):
    def test_panel_freezing_is_stratified_and_skips_placeholders(self) -> None:
        counts = {"alpha": 25, "beta": 25}
        rows = []
        for subject, count in counts.items():
            for index in range(count + 2):
                rows.append(
                    {
                        "question_id": f"{subject}-{index}",
                        "category": subject,
                        "question": f"Question {index}?",
                        "options": [f"option-{choice}" for choice in range(10)],
                        "answer_index": index % 10,
                    }
                )
        rows.append(
            {
                "question_id": "excluded",
                "category": "alpha",
                "question": "Excluded?",
                "options": ["x"] * 9 + ["N/A"],
                "answer_index": 0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, panel, manifest = (
                root / "source.jsonl", root / "panel.jsonl", root / "manifest.json"
            )
            write_jsonl(source, rows)
            frozen = freeze_panel(
                source, panel, manifest, "a" * 40, counts, selection_seed=44
            )
            self.assertEqual(frozen["subject_counts"], counts)
            self.assertEqual(frozen["eligible_ten_option_rows"], 54)
            self.assertEqual(len(panel.read_text().splitlines()), 50)

    def test_exact_brier_rewards_and_grids(self) -> None:
        rewards = {
            brier_utility(answer, confidence, correct)
            for answer in range(10)
            for confidence in range(9)
            for correct in range(10)
        }
        self.assertTrue(all(isinstance(value, Fraction) for value in rewards))
        self.assertGreaterEqual(min(rewards), 0)
        self.assertLessEqual(max(rewards), 1)
        self.assertEqual(len(exact_return_grid(2)), 106)
        self.assertEqual(len(exact_return_grid(4)), 552)
        self.assertEqual(len(exact_return_grid(6)), 883)

    def test_shared_and_untied_population_targets_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mock.json"
            artifact = write_mock_kernel_artifact(path, 1)
            self.assertTrue(validate_kernel_artifact(path)["passed"])
            for horizon in (2, 4):
                shared = shared_question_model(artifact["questions"][0], horizon)
                untied = untied_question_model(shared)
                shared_bundle = compute_influences(shared, 0.1)
                untied_bundle = compute_influences(untied, 0.1)
                self.assertAlmostEqual(
                    shared_bundle.solution.cvar,
                    untied_bundle.solution.cvar,
                    places=12,
                )
                if horizon == 2:
                    np.testing.assert_allclose(
                        solve_probability_form(shared)[horizon, 0],
                        shared_bundle.solution.probabilities,
                        atol=1e-10,
                    )
                self.assertEqual(untied.group_count, 1 + 90 * (horizon - 1))

    def test_mock_run_is_never_paper_evidence(self) -> None:
        repository = Path.cwd()
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            root = Path(temporary)
            artifact_path = root / "mock.json"
            write_mock_kernel_artifact(artifact_path, 1)
            config = {
                "schema_version": 1,
                "master_seed": 10,
                "replications": 1,
                "horizons": [2],
                "alphas": [0.1],
                "budgets_per_shared_group": [10],
                "methods": ["uniform", "tis", "untied_tis"],
                "pilot_multiplier": 4.0,
                "minimum_pilot_per_group": 1,
                "minimum_main_per_group": 2,
                "exploration_exponent": 0.25,
                "output_directory": str(root / "result"),
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = run_llm_study(config_path, artifact_path)
            manifest = json.loads((result / "manifest.json").read_text())
            self.assertEqual(
                manifest["result_evidence_status"], "execution_or_partial_only"
            )
            with (result / "raw.csv").open(encoding="utf-8") as stream:
                self.assertEqual(sum(1 for _ in stream) - 1, 3)
            self.assertTrue(verify_llm_result_directory(result)["passed"])


if __name__ == "__main__":
    unittest.main()
