"""Exact 72-outcome kernel calibration and generation audit (GPU).

One forward pass gives the label distribution over A-H, a second batched
pass conditioned on each label gives the confidence distribution over
1-9; their product is the exact 72-outcome law of the constrained
decoder. The audit recomputes selected cells through a second
computational path, draws real multinomial samples, and compares counts
with the calibration law by Bonferroni-corrected exact binomial tests.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..llm_study.common import file_sha256, object_sha256, read_json, read_jsonl, write_json
from . import CANDIDATE_LABELS, CONFIDENCE_LABELS, STATES, WORKFLOWS
from .protocol import action_for_state, render_messages

FAMILY_ALPHA = 1e-4


def _single_token_ids(tokenizer, labels: Sequence[str]) -> list[int]:
    ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Label {label!r} is not a single token")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise RuntimeError("Labels do not have distinct token identifiers")
    return ids


def _restricted_joint(model, tokenizer, messages, first_ids, second_ids, temperature):
    import torch

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    # logits_to_keep=1 materialises only the final position's logits: the
    # same computation, restricted output, which keeps the conditional
    # batch within small-GPU memory. Fall back for models without it.
    def _forward(**kwargs):
        try:
            return model(**kwargs, logits_to_keep=1)
        except TypeError:
            return model(**kwargs)

    with torch.inference_mode():
        first = _forward(**encoded, use_cache=False)
        first_logits = first.logits[0, -1, list(first_ids)].float()
        first_probability = torch.softmax(first_logits / temperature, dim=0)
        input_ids = encoded["input_ids"].repeat(len(first_ids), 1)
        appended = torch.tensor(first_ids, device=device).reshape(-1, 1)
        conditional = torch.cat((input_ids, appended), dim=1)
        second = _forward(
            input_ids=conditional,
            attention_mask=torch.ones_like(conditional),
            use_cache=False,
        )
        second_logits = second.logits[:, -1, list(second_ids)].float()
        second_probability = torch.softmax(second_logits / temperature, dim=1)
        joint = first_probability[:, None] * second_probability
    result = joint.detach().cpu().double().numpy().reshape(-1)
    result /= result.sum()
    expected = len(first_ids) * len(second_ids)
    if result.shape != (expected,) or np.any(result <= 0) or not np.all(np.isfinite(result)):
        raise RuntimeError("Invalid restricted joint law")
    return result, int(encoded["input_ids"].shape[1])


def calibrate_kernels(
    protocol_path: Path,
    panel_path: Path,
    banks_path: Path,
    output_path: Path,
    workflow: str,
    local_files_only: bool = False,
) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown workflow {workflow!r}")
    protocol = read_json(protocol_path)
    spec = protocol["generator"]
    temperature = float(protocol["decoder"]["temperature"])
    panel = read_jsonl(panel_path)
    banks_artifact = read_json(banks_path)
    banks = {int(b["panel_index"]): b["candidates"] for b in banks_artifact["banks"]}

    tokenizer = AutoTokenizer.from_pretrained(
        str(spec["tokenizer_identifier"]), revision=str(spec["tokenizer_revision"]),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(spec["identifier"]), revision=str(spec["revision"]),
        torch_dtype=torch.bfloat16, device_map=str(spec["device_map"]),
        local_files_only=local_files_only,
    )
    model.eval()
    torch.manual_seed(int(protocol["seeds"]["calibration"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(protocol["seeds"]["calibration"]))
        torch.backends.cuda.matmul.allow_tf32 = False

    candidate_ids = _single_token_ids(tokenizer, CANDIDATE_LABELS)
    confidence_ids = _single_token_ids(tokenizer, CONFIDENCE_LABELS)
    started = time.perf_counter()
    questions = []
    total_input = 0
    for item in panel:
        bank = banks[int(item["panel_index"])]
        kernels = []
        for state in range(STATES):
            messages = render_messages(item, bank, state, workflow)
            law, input_tokens = _restricted_joint(
                model, tokenizer, messages, candidate_ids, confidence_ids, temperature
            )
            total_input += input_tokens
            kernels.append(
                {
                    "state_index": state,
                    "action": action_for_state(state),
                    "input_tokens": input_tokens,
                    "probabilities": law.tolist(),
                }
            )
        questions.append(
            {
                "panel_index": int(item["panel_index"]),
                "item_id": str(item["item_id"]),
                "gold_answer": float(item["gold_answer"]),
                "kernels": kernels,
            }
        )
    artifact: dict[str, object] = {
        "schema_version": 1,
        "evidence_status": "paper_eligible",
        "study": "finqa_terminal_risk",
        "workflow": workflow,
        "protocol_sha256": file_sha256(protocol_path),
        "panel_file_sha256": file_sha256(panel_path),
        "candidate_banks_sha256": file_sha256(banks_path),
        "generator": dict(spec),
        "temperature": temperature,
        "candidate_labels": list(CANDIDATE_LABELS),
        "candidate_token_ids": candidate_ids,
        "confidence_labels": list(CONFIDENCE_LABELS),
        "confidence_token_ids": confidence_ids,
        "software": {"torch": torch.__version__, "transformers": transformers.__version__},
        "calibration_cost": {
            "wall_seconds": time.perf_counter() - started,
            "prompt_kernels": len(panel) * STATES,
            "model_forward_calls": 2 * len(panel) * STATES,
            "input_tokens_across_unique_prompts": total_input,
        },
        "questions": questions,
    }
    artifact["content_sha256"] = object_sha256(artifact)
    write_json(output_path, artifact)
    return artifact


def _binomial_two_sided_p(count: int, n: int, p: float) -> float:
    """Exact two-sided binomial p-value, shared with the MMLU-study audit.

    Works in log space (doubled smaller tail), exact at any sample count;
    degenerate cells whose enumerated probability underflows are scored by
    the deterministic law directly.
    """
    if p <= 0.0:
        return 1.0 if count == 0 else 0.0
    if p >= 1.0:
        return 1.0 if count == n else 0.0
    from ..llm_study.audit import _binomial_two_sided_p as shared_binomial_p

    return shared_binomial_p(count, n, p)


def audit_generations(
    artifact_path: Path,
    protocol_path: Path,
    panel_path: Path,
    banks_path: Path,
    output_path: Path,
    questions: int = 3,
    state_indices: Sequence[int] = (0, 14, 40, 58, 72),
    samples: int = 2048,
    batch_size: int = 128,
    local_files_only: bool = False,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    artifact = read_json(artifact_path)
    protocol = read_json(protocol_path)
    spec = protocol["generator"]
    temperature = float(protocol["decoder"]["temperature"])
    panel = {int(i["panel_index"]): i for i in read_jsonl(panel_path)}
    banks = {
        int(b["panel_index"]): b["candidates"]
        for b in read_json(banks_path)["banks"]
    }
    workflow = str(artifact["workflow"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec["tokenizer_identifier"]), revision=str(spec["tokenizer_revision"]),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(spec["identifier"]), revision=str(spec["revision"]),
        torch_dtype=torch.bfloat16, device_map=str(spec["device_map"]),
        local_files_only=local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device
    candidate_ids = artifact["candidate_token_ids"]
    confidence_ids = artifact["confidence_token_ids"]
    audited = artifact["questions"][:questions]
    cells = len(audited) * len(state_indices) * 72
    per_cell_alpha = FAMILY_ALPHA / cells
    records = []
    smallest = 1.0
    started = time.perf_counter()
    torch.manual_seed(int(protocol["seeds"]["audit"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(protocol["seeds"]["audit"]))
    for question in audited:
        item = panel[int(question["panel_index"])]
        bank = banks[int(question["panel_index"])]
        laws = {int(k["state_index"]): np.asarray(k["probabilities"]) for k in question["kernels"]}
        for state in state_indices:
            messages = render_messages(item, bank, state, workflow)
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            counts = np.zeros(72, dtype=int)
            remaining = samples
            def _forward(**kwargs):
                try:
                    return model(**kwargs, logits_to_keep=1)
                except TypeError:
                    return model(**kwargs)

            with torch.inference_mode():
                base = _forward(**encoded, use_cache=True)
                first_logits = base.logits[0, -1, list(candidate_ids)].float()
                first_probability = torch.softmax(first_logits / temperature, dim=0)
                # conditional confidence laws per candidate, cached prefill
                second_probability = []
                for ci in candidate_ids:
                    ids = torch.cat(
                        (encoded["input_ids"], torch.tensor([[ci]], device=device)), dim=1
                    )
                    out = _forward(
                        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                    )
                    logits = out.logits[0, -1, list(confidence_ids)].float()
                    second_probability.append(torch.softmax(logits / temperature, dim=0))
                second_probability = torch.stack(second_probability)
                generation_law = (
                    (first_probability[:, None] * second_probability).cpu().double().numpy().reshape(-1)
                )
                generation_law /= generation_law.sum()
                sampler = torch.distributions.Categorical(
                    probs=torch.tensor(generation_law, device=device)
                )
                while remaining > 0:
                    draw = sampler.sample((min(batch_size, remaining),))
                    values, freq = torch.unique(draw, return_counts=True)
                    counts[values.cpu().numpy()] += freq.cpu().numpy()
                    remaining -= int(draw.shape[0])
            enumerated = laws[state]
            failures = 0
            worst = 1.0
            for outcome in range(72):
                p_value = _binomial_two_sided_p(int(counts[outcome]), samples, float(enumerated[outcome]))
                worst = min(worst, p_value)
                if p_value < per_cell_alpha:
                    generation_p = _binomial_two_sided_p(
                        int(counts[outcome]), samples, float(generation_law[outcome])
                    )
                    verdict = "precision_gap" if generation_p >= per_cell_alpha else "sampler_fault"
                    failures += 1
                    records.append(
                        {
                            "panel_index": int(question["panel_index"]),
                            "state_index": int(state),
                            "outcome": outcome,
                            "count": int(counts[outcome]),
                            "enumerated_p": float(enumerated[outcome]),
                            "generation_path_p": float(generation_law[outcome]),
                            "p_value": p_value,
                            "verdict": verdict,
                        }
                    )
            smallest = min(smallest, worst)
    passed = all(r["verdict"] != "sampler_fault" for r in records)
    report = {
        "schema_version": 1,
        "study": "finqa_terminal_risk",
        "workflow": workflow,
        "kernel_artifact_sha256": file_sha256(artifact_path),
        "audited_cells": cells,
        "samples_per_prompt": samples,
        "family_alpha": FAMILY_ALPHA,
        "per_cell_alpha": per_cell_alpha,
        "smallest_p_value": smallest,
        "records": records,
        "passed": passed,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_path, report)
    return report
