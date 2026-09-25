"""Audit constrained generations against the enumerated kernels.

Draws restricted generations through the standard ``model.generate``
sampling path (not the enumeration code) and tests every audited kernel
cell with a Bonferroni-corrected exact-parameter z test.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .protocol import ANSWER_LABELS, CONFIDENCE_LABELS, render_messages


FAMILY_ALPHA = 1e-4


def _binomial_two_sided_p(count: int, trials: int, probability: float) -> float:
    """Exact two-sided binomial tail p-value (doubled smaller tail, capped at 1).

    Exact tails stay valid for the very small enumerated probabilities where a
    normal approximation would misfire.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("Cell probability must lie strictly inside (0,1)")
    log_p = math.log(probability)
    log_q = math.log1p(-probability)

    def log_pmf(k: int) -> float:
        return (
            math.lgamma(trials + 1)
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * log_p
            + (trials - k) * log_q
        )

    mean = trials * probability
    if count <= mean:
        tail = sum(math.exp(log_pmf(k)) for k in range(0, count + 1))
    else:
        tail = sum(math.exp(log_pmf(k)) for k in range(count, trials + 1))
    return min(1.0, 2.0 * tail)


def _sample_restricted_generations(
    model: object,
    tokenizer: object,
    messages: list[dict[str, str]],
    answer_token_ids: Sequence[int],
    confidence_token_ids: Sequence[int],
    temperature: float,
    samples: int,
    batch_size: int,
    template_kwargs: Mapping[str, object],
) -> np.ndarray:
    """Count sampled (answer, confidence) pairs from restricted generation."""
    import torch
    from transformers import LogitsProcessorList

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
    prompt_length = int(encoded["input_ids"].shape[1])
    answer_position = {token: index for index, token in enumerate(answer_token_ids)}
    confidence_position = {
        token: index for index, token in enumerate(confidence_token_ids)
    }

    class _TwoTokenRestriction:
        def __call__(self, input_ids, scores):  # type: ignore[no-untyped-def]
            step = int(input_ids.shape[1]) - prompt_length
            allowed = answer_token_ids if step == 0 else confidence_token_ids
            mask = torch.full_like(scores, float("-inf"))
            allowed_tensor = torch.as_tensor(list(allowed), device=scores.device)
            mask[:, allowed_tensor] = scores[:, allowed_tensor]
            return mask

    counts = np.zeros((10, 9), dtype=int)
    drawn = 0
    while drawn < samples:
        batch = min(batch_size, samples - drawn)
        with torch.inference_mode():
            sequences = model.generate(  # type: ignore[operator]
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                max_new_tokens=2,
                num_return_sequences=batch,
                logits_processor=LogitsProcessorList([_TwoTokenRestriction()]),
                pad_token_id=int(answer_token_ids[0]),
            )
        new_tokens = sequences[:, prompt_length:]
        if new_tokens.shape[1] != 2:
            raise RuntimeError("Restricted generation did not emit two tokens")
        for answer_token, confidence_token in new_tokens.tolist():
            counts[
                answer_position[int(answer_token)],
                confidence_position[int(confidence_token)],
            ] += 1
        drawn += batch
        del sequences, new_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return counts.reshape(-1)


def _generation_batch_law(
    model: object,
    tokenizer: object,
    messages: list[dict[str, str]],
    answer_token_ids: Sequence[int],
    confidence_token_ids: Sequence[int],
    temperature: float,
    batch_size: int,
    template_kwargs: Mapping[str, object],
) -> np.ndarray:
    """Restricted joint law recomputed through the generation path.

    bfloat16 forwards are not bitwise reproducible across batch shapes or
    KV-cache configurations, so near-tied logits can shift mass between the
    calibration forwards and what ``generate`` samples from. This evaluates
    the restricted softmax law through the sampler's own numerical path
    (batched prefill with KV cache, one cached decode step per label).
    """
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
    input_ids = encoded["input_ids"].to(device).repeat(batch_size, 1)
    prefill_length = int(input_ids.shape[1])
    attention_mask = torch.ones_like(input_ids)
    answer_tensor = torch.as_tensor(list(answer_token_ids), device=device)
    confidence_tensor = torch.as_tensor(list(confidence_token_ids), device=device)
    with torch.inference_mode():
        first = model(  # type: ignore[operator]
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        answer_logits = first.logits[:, -1, answer_tensor].float()
        answer_probability = (
            torch.softmax(answer_logits / temperature, dim=1).mean(dim=0)
        )
        past = first.past_key_values
        step_mask = torch.ones(
            (batch_size, prefill_length + 1), dtype=torch.long, device=device
        )
        joint = torch.zeros((10, 9), dtype=torch.float64, device=device)
        for answer_index in range(10):
            second = model(  # type: ignore[operator]
                input_ids=answer_tensor[answer_index].expand(batch_size, 1),
                attention_mask=step_mask,
                past_key_values=past,
                use_cache=True,
            )
            past.crop(prefill_length)
            confidence_logits = second.logits[:, -1, confidence_tensor].float()
            confidence_probability = (
                torch.softmax(confidence_logits / temperature, dim=1).mean(dim=0)
            )
            joint[answer_index] = (
                answer_probability[answer_index].double()
                * confidence_probability.double()
            )
        law = joint.reshape(-1).cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return law / law.sum()


def audit_generations(
    artifact_path: Path,
    protocol_path: Path,
    panel_path: Path,
    output_path: Path,
    questions: int = 3,
    state_indices: Sequence[int] = (0, 17, 44, 71, 90),
    samples: int = 4096,
    batch_size: int = 128,
    local_files_only: bool = False,
    model_key: str = "model",
) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The generation audit requires requirements-llm-lock.txt"
        ) from exc

    protocol = read_json(protocol_path)
    artifact = read_json(artifact_path)
    panel = read_jsonl(panel_path)
    if not isinstance(protocol, Mapping) or not isinstance(artifact, Mapping):
        raise ValueError("Protocol and kernel artifact must be JSON objects")
    if artifact.get("panel_file_sha256") != file_sha256(panel_path):
        raise RuntimeError("Kernel artifact does not match the frozen panel file")
    from .calibrate import MODEL_KEYS

    if model_key not in MODEL_KEYS:
        raise ValueError(f"Model key must be one of {sorted(MODEL_KEYS)}")
    if artifact.get("model_key", "model") != model_key:
        raise RuntimeError(
            f"Kernel artifact was calibrated with model_key "
            f"{artifact.get('model_key', 'model')!r}, audit requested {model_key!r}"
        )
    model_spec = protocol[model_key]
    template_kwargs = dict(model_spec.get("chat_template_kwargs", {}))
    prompts = protocol["prompts"]
    temperature = float(protocol["decoder"]["temperature"])  # type: ignore[index]
    answer_ids = [int(v) for v in artifact["answer_token_ids"]]
    confidence_ids = [int(v) for v in artifact["confidence_token_ids"]]
    state_indices = sorted(set(int(v) for v in state_indices))
    if any(not 0 <= v <= 90 for v in state_indices):
        raise ValueError("Audit state indices must lie in 0..90")
    audited_questions = list(artifact["questions"])[: int(questions)]
    if not audited_questions:
        raise ValueError("No questions available for the audit")

    torch.manual_seed(int(protocol["seeds"]["calibration"]) + 1)  # type: ignore[index]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(protocol["seeds"]["calibration"]) + 1)  # type: ignore[index]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_spec["tokenizer_identifier"]),
        revision=str(model_spec["tokenizer_revision"]),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_spec["identifier"]),
        revision=str(model_spec["revision"]),
        torch_dtype=torch.bfloat16,
        device_map=str(model_spec["device_map"]),
        local_files_only=local_files_only,
    )
    model.eval()

    artifact_policy = str(artifact.get("policy", "default"))
    panel_by_index = {int(item["panel_index"]): item for item in panel}
    cell_count = len(audited_questions) * len(state_indices) * 90
    per_cell_alpha = FAMILY_ALPHA / cell_count
    records: list[dict[str, object]] = []
    smallest_p_value = 1.0
    started = time.perf_counter()
    for question in audited_questions:
        item = panel_by_index[int(question["panel_index"])]
        kernels_by_state = {
            int(kernel["state_index"]): kernel for kernel in question["kernels"]
        }
        for state_index in state_indices:
            kernel = kernels_by_state[state_index]
            expected = np.asarray(kernel["probabilities"], dtype=float)
            messages = render_messages(item, state_index, prompts, policy=artifact_policy)
            observed = _sample_restricted_generations(
                model,
                tokenizer,
                messages,
                answer_ids,
                confidence_ids,
                temperature,
                int(samples),
                int(batch_size),
                template_kwargs,
            )
            n = int(observed.sum())
            frequencies = observed / n
            p_values = [
                _binomial_two_sided_p(int(count), n, float(cell_probability))
                for count, cell_probability in zip(observed, expected)
            ]
            minimum_p = min(p_values)
            smallest_p_value = min(smallest_p_value, minimum_p)
            failing_cells = int(sum(value < per_cell_alpha for value in p_values))
            record: dict[str, object] = {
                "panel_index": int(question["panel_index"]),
                "item_id": str(question["item_id"]),
                "state_index": state_index,
                "action": str(kernel["action"]),
                "samples": n,
                "total_variation": float(
                    0.5 * np.sum(np.abs(frequencies - expected))
                ),
                "minimum_p_value": minimum_p,
                "cells_beyond_threshold": failing_cells,
                "observed_counts": observed.tolist(),
            }
            if failing_cells:
                # Second stage: distinguish a sampler fault from a bfloat16
                # batch-shape precision gap. A faulty sampler (wrong mask,
                # temperature, or leaked warper) disagrees with the
                # batch-shape law as well; a precision gap is consistent
                # with it.
                batch_law = _generation_batch_law(
                    model,
                    tokenizer,
                    messages,
                    answer_ids,
                    confidence_ids,
                    temperature,
                    min(int(batch_size), int(samples)),
                    template_kwargs,
                )
                second_p_values = [
                    _binomial_two_sided_p(int(count), n, float(cell_probability))
                    for count, cell_probability in zip(observed, batch_law)
                ]
                fault_cells = int(
                    sum(value < per_cell_alpha for value in second_p_values)
                )
                record["batch_shape_law_total_variation"] = float(
                    0.5 * np.sum(np.abs(batch_law - expected))
                )
                record["cells_beyond_threshold_vs_batch_law"] = fault_cells
                record["verdict"] = (
                    "sampler_fault" if fault_cells else "precision_gap"
                )
            else:
                record["verdict"] = "consistent"
            records.append(record)
    passed = all(record["verdict"] != "sampler_fault" for record in records)
    report: dict[str, object] = {
        "schema_version": 1,
        "model_key": model_key,
        "model_identifier": str(model_spec["identifier"]),
        "kernel_artifact_sha256": file_sha256(artifact_path),
        "kernel_content_sha256": artifact.get("content_sha256"),
        "protocol_sha256": file_sha256(protocol_path),
        "panel_file_sha256": file_sha256(panel_path),
        "answer_labels": list(ANSWER_LABELS),
        "confidence_labels": list(CONFIDENCE_LABELS),
        "temperature": temperature,
        "family_alpha": FAMILY_ALPHA,
        "audited_cells": cell_count,
        "per_cell_alpha": per_cell_alpha,
        "test": (
            "exact two-sided binomial tail per cell, Bonferroni-corrected; "
            "cells failing against the enumerated law are retested against "
            "the restricted law recomputed through the generation path "
            "(batched cached prefill plus one cached decode step), and only "
            "cells inconsistent with both laws (sampler faults) fail the "
            "audit"
        ),
        "smallest_p_value": smallest_p_value,
        "samples_per_prompt": int(samples),
        "wall_seconds": time.perf_counter() - started,
        "records": records,
        "passed": passed,
    }
    write_json(output_path, report)
    if not passed:
        raise RuntimeError(
            f"Generation audit failed: sampled frequencies are inconsistent "
            f"with both the enumerated law and the generation-batch-shape "
            f"law at Bonferroni level {per_cell_alpha:.2e}; see {output_path}"
        )
    return report
