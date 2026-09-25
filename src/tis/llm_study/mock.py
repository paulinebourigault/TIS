from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import object_sha256, write_json
from .protocol import action_for_state


def write_mock_kernel_artifact(path: Path, questions: int = 2) -> dict[str, object]:
    """Write a deterministic CI fixture that is never valid paper evidence."""
    if not 1 <= questions <= 5:
        raise ValueError("Mock fixture is intentionally limited to one through five questions")
    records: list[dict[str, object]] = []
    for question_index in range(questions):
        kernels: list[dict[str, object]] = []
        for state_index in range(91):
            rng = np.random.default_rng(
                np.random.SeedSequence([9901, question_index, state_index])
            )
            concentration = np.linspace(0.8, 1.6, 90)
            concentration = np.roll(concentration, 9 * question_index + state_index)
            probability = rng.dirichlet(concentration)
            kernels.append(
                {
                    "state_index": state_index,
                    "action": action_for_state(state_index),
                    "prompt_sha256": f"mock-{question_index:02d}-{state_index:02d}",
                    "input_tokens": 0,
                    "probabilities": probability.tolist(),
                }
            )
        records.append(
            {
                "panel_index": question_index,
                "item_id": f"MOCK-{question_index:03d}",
                "subject": "mock-only",
                "correct_index": question_index % 10,
                "kernels": kernels,
            }
        )
    artifact: dict[str, object] = {
        "schema_version": 1,
        "evidence_status": "CI_ONLY_NOT_PAPER_EVIDENCE",
        "protocol_sha256": "mock",
        "panel_file_sha256": "mock",
        "panel_manifest_sha256": "mock",
        "model": {"identifier": "deterministic-dirichlet-mock"},
        "temperature": 1.0,
        "questions": records,
    }
    artifact["content_sha256"] = object_sha256(artifact)
    write_json(path, artifact)
    return artifact

