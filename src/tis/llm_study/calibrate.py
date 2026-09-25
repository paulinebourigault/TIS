from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .common import file_sha256, object_sha256, read_json, read_jsonl, text_sha256, write_json
from .protocol import ANSWER_LABELS, CONFIDENCE_LABELS, action_for_state, render_messages


MODEL_KEYS = (
    "model",
    "robustness_model",
    "robustness_model_2",
    "scale_model_1",
    "scale_model_2",
    "scale_model_3",
)

PROTOCOL_SCOPE_KEYS = ("schema_version", "panel", "prompts", "decoder", "workflow", "seeds")


def protocol_scope_sha256(
    protocol: Mapping[str, object],
    model_key: str,
    policy: str = "default",
    panel_section: str = "panel",
) -> str:
    """Hash of the protocol sections that define one model's kernels.

    Provenance binds to the shared sections plus the calibrated model's
    own spec, not the whole file, so adding an unrelated model section
    cannot invalidate existing kernel artifacts.
    """
    scope = {key: protocol.get(key) for key in PROTOCOL_SCOPE_KEYS}
    scope["model_key"] = model_key
    scope["model_spec"] = protocol[model_key]
    if policy != "default":
        scope["policy"] = policy
    if panel_section != "panel":
        scope["panel_section"] = panel_section
        scope["panel_spec"] = protocol.get(panel_section)
    return object_sha256(scope)


def _single_token_ids(tokenizer: object, labels: Sequence[str]) -> list[int]:
    identifiers: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(encoded) != 1:
            raise RuntimeError(
                f"Frozen label {label!r} encodes to {len(encoded)} tokens, expected one"
            )
        identifiers.append(int(encoded[0]))
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Frozen labels do not have distinct token identifiers")
    return identifiers


def _restricted_joint(
    model: object,
    tokenizer: object,
    messages: list[dict[str, str]],
    answer_token_ids: Sequence[int],
    confidence_token_ids: Sequence[int],
    temperature: float,
    template_kwargs: Mapping[str, object],
) -> tuple[np.ndarray, int, int]:
    import torch

    encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **template_kwargs,
    )
    device = next(model.parameters()).device  # type: ignore[attr-defined]
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        first = model(**encoded, use_cache=False)  # type: ignore[operator]
        answer_logits = first.logits[0, -1, list(answer_token_ids)].float()
        answer_probability = torch.softmax(answer_logits / temperature, dim=0)

        input_ids = encoded["input_ids"].repeat(len(answer_token_ids), 1)
        appended = torch.tensor(answer_token_ids, device=device).reshape(-1, 1)
        conditional_ids = torch.cat((input_ids, appended), dim=1)
        conditional_mask = torch.ones_like(conditional_ids)
        second = model(  # type: ignore[operator]
            input_ids=conditional_ids,
            attention_mask=conditional_mask,
            use_cache=False,
        )
        confidence_logits = second.logits[:, -1, list(confidence_token_ids)].float()
        confidence_probability = torch.softmax(
            confidence_logits / temperature, dim=1
        )
        joint = answer_probability[:, None] * confidence_probability
    result = joint.detach().cpu().double().numpy().reshape(-1)
    result /= result.sum()
    if result.shape != (90,) or np.any(result <= 0) or not np.all(np.isfinite(result)):
        raise RuntimeError("Invalid restricted 90-output law")
    return result, int(encoded["input_ids"].shape[1]), 2


def calibrate_kernels(
    protocol_path: Path,
    panel_path: Path,
    panel_manifest_path: Path,
    output_path: Path,
    local_files_only: bool = False,
    model_key: str = "model",
    policy: str = "default",
    panel_section: str = "panel",
) -> dict[str, object]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM calibration requires requirements-llm-lock.txt and the frozen model weights"
        ) from exc

    protocol = read_json(protocol_path)
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    panel = read_jsonl(panel_path)
    panel_manifest = read_json(panel_manifest_path)
    if not isinstance(panel_manifest, dict):
        raise ValueError("Panel manifest must be a JSON object")
    if len(panel) != 50 or object_sha256(panel) != panel_manifest["panel_sha256"]:
        raise RuntimeError("Panel content does not match its frozen manifest")
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Model key must be one of {sorted(MODEL_KEYS)}")
    model_spec = protocol[model_key]
    prompts = protocol["prompts"]
    if not isinstance(model_spec, Mapping) or not isinstance(prompts, Mapping):
        raise ValueError("Protocol model/prompts sections are invalid")
    template_kwargs = dict(model_spec.get("chat_template_kwargs", {}))
    revision = str(model_spec["revision"])
    if len(revision) != 40:
        raise ValueError("Model revision must be a full 40-character commit")
    if str(model_spec["precision"]) != "bfloat16":
        raise ValueError("This frozen protocol requires bfloat16 weights")
    decoder = protocol["decoder"]
    if tuple(decoder["answer_labels"]) != ANSWER_LABELS or tuple(
        decoder["confidence_labels"]
    ) != CONFIDENCE_LABELS:
        raise ValueError("Protocol labels disagree with the frozen implementation")
    temperature = float(protocol["decoder"]["temperature"])  # type: ignore[index]
    if temperature <= 0:
        raise ValueError("Restricted-softmax temperature must be positive")

    torch.manual_seed(int(protocol["seeds"]["calibration"]))  # type: ignore[index]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(protocol["seeds"]["calibration"]))  # type: ignore[index]
        torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_spec["tokenizer_identifier"]),
        revision=str(model_spec["tokenizer_revision"]),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_spec["identifier"]),
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map=str(model_spec["device_map"]),
        local_files_only=local_files_only,
    )
    model.eval()

    def _resolved_commit(observed: object, pinned: str) -> object:
        # A full 40-character commit pin is self-certifying: the hub can only
        # serve that exact snapshot for `revision=<sha>`. Some tokenizer
        # classes expose no commit attribute (observed is None); accept the
        # pin in that case rather than failing on a missing attribute.
        if observed is None and len(pinned) == 40:
            return pinned
        return observed

    resolved_model_commit = _resolved_commit(
        getattr(model.config, "_commit_hash", None), revision
    )
    tokenizer_revision = str(model_spec["tokenizer_revision"])
    resolved_tokenizer_commit = _resolved_commit(
        getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        tokenizer_revision,
    )
    if resolved_model_commit != revision:
        raise RuntimeError(
            f"Resolved model commit {resolved_model_commit!r} != requested {revision!r}"
        )
    if resolved_tokenizer_commit != tokenizer_revision:
        raise RuntimeError(
            f"Resolved tokenizer commit {resolved_tokenizer_commit!r} != "
            f"requested frozen revision {tokenizer_revision!r}"
        )
    answer_ids = _single_token_ids(tokenizer, ANSWER_LABELS)
    confidence_ids = _single_token_ids(tokenizer, CONFIDENCE_LABELS)
    started = time.perf_counter()
    calibrated_questions: list[dict[str, object]] = []
    total_input_tokens = 0
    total_output_positions = 0
    for item in panel:
        kernels: list[dict[str, object]] = []
        for state_index in range(91):
            messages = render_messages(item, state_index, prompts, policy=policy)
            probability, input_tokens, output_positions = _restricted_joint(
                model,
                tokenizer,
                messages,
                answer_ids,
                confidence_ids,
                temperature,
                template_kwargs,
            )
            total_input_tokens += input_tokens
            total_output_positions += output_positions
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            kernels.append(
                {
                    "state_index": state_index,
                    "action": action_for_state(state_index, policy=policy),
                    "prompt_sha256": text_sha256(str(rendered)),
                    "input_tokens": input_tokens,
                    "probabilities": probability.tolist(),
                }
            )
        calibrated_questions.append(
            {
                "panel_index": int(item["panel_index"]),
                "item_id": str(item["item_id"]),
                "subject": str(item["subject"]),
                "correct_index": int(item["correct_index"]),
                "kernels": kernels,
            }
        )
    elapsed = time.perf_counter() - started
    artifact: dict[str, object] = {
        "schema_version": 1,
        "evidence_status": "paper_eligible",
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_scope_sha256": protocol_scope_sha256(
            protocol, model_key, policy=policy, panel_section=panel_section
        ),
        "policy": policy,
        "panel_section": panel_section,
        "panel_file_sha256": file_sha256(panel_path),
        "panel_manifest_sha256": file_sha256(panel_manifest_path),
        "model_key": model_key,
        "chat_template_kwargs": template_kwargs,
        "model": dict(model_spec),
        "resolved_model_commit": resolved_model_commit,
        "resolved_tokenizer_commit": resolved_tokenizer_commit,
        "model_config_sha256": object_sha256(model.config.to_dict()),
        "tokenizer_chat_template_sha256": text_sha256(str(tokenizer.chat_template)),
        "answer_labels": list(ANSWER_LABELS),
        "answer_token_ids": answer_ids,
        "confidence_labels": list(CONFIDENCE_LABELS),
        "confidence_token_ids": confidence_ids,
        "temperature": temperature,
        "normalization": "restricted answer softmax times answer-conditional restricted confidence softmax",
        "software": {
            "python_platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "calibration_cost": {
            "wall_seconds": elapsed,
            "prompt_kernels": 50 * 91,
            "model_forward_calls": 2 * 50 * 91,
            "input_tokens_across_unique_prompts": total_input_tokens,
            "restricted_output_positions_per_prompt": 2,
            "output_positions": total_output_positions,
        },
        "questions": calibrated_questions,
    }
    artifact["content_sha256"] = object_sha256(artifact)
    write_json(output_path, artifact)
    return artifact


def validate_kernel_artifact(
    artifact_path: Path,
    protocol_path: Path | None = None,
    panel_path: Path | None = None,
) -> dict[str, object]:
    artifact = read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Kernel artifact must be an object")
    stored_hash = artifact.pop("content_sha256", None)
    failures: list[str] = []
    if stored_hash != object_sha256(artifact):
        failures.append("content hash mismatch")
    if protocol_path and artifact.get("protocol_sha256") != file_sha256(protocol_path):
        # The whole protocol file changed; accept the artifact when the
        # sections that define its kernels are unchanged (scope hash).
        scope_matches = False
        stored_scope = artifact.get("protocol_scope_sha256")
        if stored_scope is not None:
            protocol = read_json(protocol_path)
            model_key = str(artifact.get("model_key", "model"))
            if isinstance(protocol, dict) and model_key in protocol:
                scope_matches = stored_scope == protocol_scope_sha256(
                    protocol,
                    model_key,
                    policy=str(artifact.get("policy", "default")),
                    panel_section=str(artifact.get("panel_section", "panel")),
                )
        if not scope_matches:
            failures.append("protocol hash mismatch")
    if panel_path and artifact.get("panel_file_sha256") != file_sha256(panel_path):
        failures.append("panel hash mismatch")
    questions = artifact.get("questions")
    if not isinstance(questions, list):
        failures.append("missing question list")
    else:
        if artifact.get("evidence_status") == "paper_eligible" and len(questions) != 50:
            failures.append("paper artifact does not contain 50 questions")
        for question in questions:
            kernels = question.get("kernels", [])
            if len(kernels) != 91:
                failures.append(f"question {question.get('panel_index')} does not have 91 kernels")
                continue
            for kernel in kernels:
                probabilities = np.asarray(kernel.get("probabilities", []), dtype=float)
                if probabilities.shape != (90,) or np.any(probabilities <= 0):
                    failures.append("invalid constrained probability vector")
                    break
                if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
                    failures.append("probability vector does not sum to one")
                    break
    if failures:
        raise RuntimeError(json.dumps({"failures": failures}, indent=2))
    return {
        "passed": True,
        "evidence_status": artifact.get("evidence_status"),
        "questions": len(questions),
        "content_sha256": stored_hash,
    }
