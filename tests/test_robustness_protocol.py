"""The declared robustness checkpoints must mirror the primary protocol."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent

ROBUSTNESS_RUNS = {
    "robustness_model": (
        "llm_study_paper_robustness.json",
        "results/llm_study_paper_robustness",
    ),
    "robustness_model_2": (
        "llm_study_paper_robustness2.json",
        "results/llm_study_paper_robustness2",
    ),
}

SCALE_RUNS = {
    "scale_model_1": (
        "llm_study_paper_scale1.json",
        "results/llm_study_paper_scale1",
    ),
    "scale_model_2": (
        "llm_study_paper_scale2.json",
        "results/llm_study_paper_scale2",
    ),
    "scale_model_3": (
        "llm_study_paper_scale3.json",
        "results/llm_study_paper_scale3",
    ),
}


class RobustnessProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(
            (ROOT / "configs" / "llm_study_protocol.json").read_text()
        )
        self.primary_config = json.loads(
            (ROOT / "configs" / "llm_study_paper.json").read_text()
        )

    def test_robustness_models_are_pinned_and_pairwise_distinct_families(
        self,
    ) -> None:
        primary = self.protocol["model"]
        specs = [primary] + [
            self.protocol[key] for key in list(ROBUSTNESS_RUNS) + list(SCALE_RUNS)
        ]
        for spec in specs:
            self.assertEqual(len(spec["revision"]), 40)
            int(spec["revision"], 16)
            self.assertEqual(spec["revision"], spec["tokenizer_revision"])
            self.assertEqual(spec["precision"], "bfloat16")
            self.assertEqual(spec["inference_library"], primary["inference_library"])
        # Prespecified generators must come from pairwise distinct families;
        # scale replications may deliberately reuse the primary family.
        prespecified = [primary] + [self.protocol[key] for key in ROBUSTNESS_RUNS]
        families = [spec["identifier"].split("/")[0] for spec in prespecified]
        self.assertEqual(len(set(families)), len(families))

    def test_scale_models_declare_post_hoc_role_and_thinking_controls(
        self,
    ) -> None:
        for key in SCALE_RUNS:
            spec = self.protocol[key]
            self.assertIn("post-hoc", spec["role"])
            self.assertIn("not part of the prespecified", spec["role"])
        self.assertEqual(
            self.protocol["scale_model_2"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_thinking_model_disables_thinking_in_frozen_template_kwargs(
        self,
    ) -> None:
        granite = self.protocol["robustness_model_2"]
        self.assertEqual(
            granite["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_robustness_run_configs_differ_only_in_output_directory(self) -> None:
        primary = dict(self.primary_config)
        self.assertEqual(primary.pop("output_directory"), "results/llm_study_paper")
        for key, (config_name, output_directory) in {
            **ROBUSTNESS_RUNS,
            **SCALE_RUNS,
        }.items():
            robustness = json.loads(
                (ROOT / "configs" / config_name).read_text()
            )
            self.assertEqual(
                robustness.pop("output_directory"), output_directory, key
            )
            self.assertEqual(primary, robustness, key)

    def test_model_keys_cover_every_protocol_model_section(self) -> None:
        from tis.llm_study.calibrate import MODEL_KEYS

        protocol_model_sections = {
            key
            for key in self.protocol
            if key == "model"
            or key.startswith(("robustness_model", "scale_model"))
        }
        self.assertEqual(set(MODEL_KEYS), protocol_model_sections)


if __name__ == "__main__":
    unittest.main()
