"""Adding a model section to the protocol must not invalidate artifacts."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tis.llm_study.calibrate import (  # noqa: E402
    protocol_scope_sha256,
    validate_kernel_artifact,
)
from tis.llm_study.common import file_sha256, object_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _minimal_artifact(protocol_path: Path, protocol: dict) -> dict:
    artifact = {
        "schema_version": 1,
        "evidence_status": "CI_ONLY_NOT_PAPER_EVIDENCE",
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_scope_sha256": protocol_scope_sha256(protocol, "model"),
        "model_key": "model",
        "questions": [
            {
                "panel_index": 0,
                "kernels": [
                    {
                        "state_index": index,
                        "probabilities": [1.0 / 90.0] * 90,
                    }
                    for index in range(91)
                ],
            }
        ],
    }
    artifact["content_sha256"] = object_sha256(artifact)
    return artifact


class ProtocolScopeTests(unittest.TestCase):
    def test_unrelated_protocol_addition_keeps_artifact_valid(self) -> None:
        protocol = json.loads(
            (ROOT / "configs" / "llm_study_protocol.json").read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            protocol_path = Path(tmp) / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            artifact = _minimal_artifact(protocol_path, protocol)
            artifact_path = Path(tmp) / "artifact.json"
            artifact_path.write_text(json.dumps(artifact))
            validate_kernel_artifact(artifact_path, protocol_path)

            extended = dict(protocol)
            extended["scale_model_99"] = {"identifier": "x/y", "revision": "0" * 40}
            protocol_path.write_text(json.dumps(extended))
            validate_kernel_artifact(artifact_path, protocol_path)

    def test_changed_shared_section_invalidates_artifact(self) -> None:
        protocol = json.loads(
            (ROOT / "configs" / "llm_study_protocol.json").read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            protocol_path = Path(tmp) / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            artifact = _minimal_artifact(protocol_path, protocol)
            artifact_path = Path(tmp) / "artifact.json"
            artifact_path.write_text(json.dumps(artifact))

            changed = json.loads(json.dumps(protocol))
            changed["decoder"]["temperature"] = 0.9
            protocol_path.write_text(json.dumps(changed))
            with self.assertRaises(RuntimeError):
                validate_kernel_artifact(artifact_path, protocol_path)

    def test_changed_own_model_spec_invalidates_artifact(self) -> None:
        protocol = json.loads(
            (ROOT / "configs" / "llm_study_protocol.json").read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            protocol_path = Path(tmp) / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            artifact = _minimal_artifact(protocol_path, protocol)
            artifact_path = Path(tmp) / "artifact.json"
            artifact_path.write_text(json.dumps(artifact))

            changed = json.loads(json.dumps(protocol))
            changed["model"]["revision"] = "f" * 40
            protocol_path.write_text(json.dumps(changed))
            with self.assertRaises(RuntimeError):
                validate_kernel_artifact(artifact_path, protocol_path)


if __name__ == "__main__":
    unittest.main()
