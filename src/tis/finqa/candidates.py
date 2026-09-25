"""Frozen candidate-bank construction and screened panel selection.

Each panel question gets eight candidate solutions sampled from the
generator with per-candidate seeds (``FINAL ANSWER: <number>``); parsing
failures stay in the bank as invalid candidates (utility zero). Banks,
prompts, seeds, and token counts are frozen with hashes; the gold answer
is used only after generation, to score the frozen bank.

Screening: a bank whose candidates share one severity utility makes the
CVaR target constant, so items are scanned in seeded SHA-256 order and
admitted only when the bank's utility spread reaches the threshold.
Every scanned bank, admitted or not, is recorded in a scan file.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..llm_study.common import (
    file_sha256,
    object_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from . import CANDIDATES
from .panel import _seeded_rank, normalize_items
from .protocol import utilities

GENERATION_TEMPERATURE = 0.8
MAX_NEW_TOKENS = 220
GENERATION_SEED_BASE = 20260914

FEWSHOT = (
    "Example question: what was the percentage change in revenue from 2007 "
    "to 2008 if revenue was $ 120 million in 2007 and $ 138 million in 2008?\n"
    "Example solution: Change is 138 - 120 = 18; percentage change is "
    "18 / 120 = 0.15, i.e. 15%. FINAL ANSWER: 15%\n"
)

PROMPT = (
    "Solve the following question about a financial report. Think step by "
    "step in at most three short sentences, then end with a line of the "
    "exact form 'FINAL ANSWER: <number>' (a percentage sign is allowed).\n\n"
    "{fewshot}\n{context}\n\nQuestion: {question}\n"
)

ANSWER_PATTERN = re.compile(
    r"FINAL ANSWER:\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*(%|percent)?",
    re.IGNORECASE,
)


def parse_answer(text: str) -> tuple[float | None, bool]:
    """Return (numeric answer, percent flag); None when unparseable."""
    matches = list(ANSWER_PATTERN.finditer(text))
    if not matches:
        return None, False
    match = matches[-1]
    value = float(match.group(1).replace(",", ""))
    return value, bool(match.group(2))


BankGenerator = Callable[[Mapping[str, object], int], dict[str, object]]


def make_bank_generator(
    spec: Mapping[str, object], local_files_only: bool = False
) -> BankGenerator:
    """Load the pinned generator; the returned callable samples one bank.

    ``generate(item, item_seed_base)`` draws the eight candidates with
    torch seeds ``item_seed_base + slot`` and returns the bank fields
    (prompt_tokens, candidates, input_tokens, output_tokens, wall_seconds).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(spec["tokenizer_identifier"]),
        revision=str(spec["tokenizer_revision"]),
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(spec["identifier"]),
        revision=str(spec["revision"]),
        torch_dtype=torch.bfloat16,
        device_map=str(spec["device_map"]),
        local_files_only=local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device

    def generate(item: Mapping[str, object], item_seed_base: int) -> dict[str, object]:
        started = time.perf_counter()
        prompt = PROMPT.format(
            fewshot=FEWSHOT, context=item["context"], question=item["question"]
        )
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])
        input_tokens = output_tokens = 0
        candidates = []
        for slot in range(CANDIDATES):
            torch.manual_seed(item_seed_base + slot)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=0.95,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, prompt_tokens:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            input_tokens += prompt_tokens
            output_tokens += int(new_tokens.shape[0])
            value, is_percent = parse_answer(text)
            derivation = text.split("FINAL ANSWER")[0].strip()
            derivation = " ".join(derivation.split())[:400]
            candidates.append(
                {
                    "slot": slot,
                    "derivation": derivation if derivation else "(no derivation)",
                    "raw_answer": value,
                    "is_percent": is_percent,
                    "answer": value,
                    "valid": value is not None,
                    "output_tokens": int(new_tokens.shape[0]),
                }
            )
        return {
            "prompt_tokens": prompt_tokens,
            "candidates": candidates,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_seconds": time.perf_counter() - started,
        }

    return generate


def build_candidate_banks(
    protocol_path: Path,
    panel_path: Path,
    output_path: Path,
    local_files_only: bool = False,
) -> dict[str, object]:
    """v1 builder: one bank per panel item, seeded by panel_index."""
    protocol = read_json(protocol_path)
    spec = protocol["generator"]
    panel = read_jsonl(panel_path)
    generate = make_bank_generator(spec, local_files_only=local_files_only)

    started = time.perf_counter()
    total_input = total_output = 0
    banks: list[dict[str, object]] = []
    for item in panel:
        seed_base = GENERATION_SEED_BASE + 1000 * int(item["panel_index"])
        bank = generate(item, seed_base)
        total_input += int(bank["input_tokens"])
        total_output += int(bank["output_tokens"])
        banks.append(
            {
                "panel_index": int(item["panel_index"]),
                "item_id": str(item["item_id"]),
                "prompt_tokens": int(bank["prompt_tokens"]),
                "candidates": bank["candidates"],
            }
        )

    artifact: dict[str, object] = {
        "schema_version": 1,
        "protocol_sha256": file_sha256(protocol_path),
        "panel_file_sha256": file_sha256(panel_path),
        "generator": dict(spec),
        "temperature": GENERATION_TEMPERATURE,
        "top_p": 0.95,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_rule": "GENERATION_SEED_BASE + 1000*panel_index + slot",
        "seed_base": GENERATION_SEED_BASE,
        "generation_cost": {
            "wall_seconds": time.perf_counter() - started,
            "input_tokens": total_input,
            "output_tokens": total_output,
        },
        "banks": banks,
    }
    artifact["content_sha256"] = object_sha256(artifact)
    write_json(output_path, artifact)
    return artifact


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    import json

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def build_screened_panel(
    protocol_path: Path,
    source_json: Path,
    panel_out: Path,
    manifest_out: Path,
    banks_out: Path,
    scan_path: Path,
    size: int,
    selection_seed: int,
    min_spread: float,
    exclude_item_ids: Sequence[str] = (),
    local_files_only: bool = False,
    generator: BankGenerator | None = None,
) -> dict[str, object]:
    """Screened panel construction (protocol v2).

    Scans eligible items in seeded SHA-256 rank order; each bank uses
    torch seeds ``GENERATION_SEED_BASE + 1000*scan_rank + slot`` and is
    admitted when its utility spread reaches ``min_spread``. Resume-safe
    via ``scan_path`` replay; ``generator`` injects a stub for tests.
    """
    protocol = read_json(protocol_path)
    items = normalize_items(source_json)
    excluded = {str(i) for i in exclude_item_ids}
    eligible = [item for item in items if item["item_id"] not in excluded]
    ordered = sorted(eligible, key=lambda it: _seeded_rank(selection_seed, it["item_id"]))
    by_id = {item["item_id"]: item for item in ordered}

    scanned: list[dict[str, object]] = []
    if scan_path.exists():
        scanned = list(read_jsonl(scan_path))
        for rank, record in enumerate(scanned):
            if int(record["scan_rank"]) != rank or (
                rank < len(ordered) and record["item_id"] != ordered[rank]["item_id"]
            ):
                raise ValueError(
                    f"{scan_path}: record {rank} does not match the seeded order; "
                    "the scan file belongs to a different selection"
                )

    accepted = [record for record in scanned if record["accepted"]]
    generate = generator
    while len(accepted) < size:
        rank = len(scanned)
        if rank >= len(ordered):
            raise ValueError(
                f"Scanned all {len(ordered)} eligible items but admitted only "
                f"{len(accepted)} of {size}"
            )
        item = ordered[rank]
        if generate is None:
            generate = make_bank_generator(
                protocol["generator"], local_files_only=local_files_only
            )
        bank = generate(item, GENERATION_SEED_BASE + 1000 * rank)
        values = utilities(bank["candidates"], float(item["gold_answer"]))
        spread = max(values) - min(values)
        record = {
            "scan_rank": rank,
            "item_id": str(item["item_id"]),
            "prompt_tokens": int(bank["prompt_tokens"]),
            "candidates": bank["candidates"],
            "utilities": values,
            "spread": spread,
            "accepted": bool(spread >= min_spread - 1e-12),
            "input_tokens": int(bank["input_tokens"]),
            "output_tokens": int(bank["output_tokens"]),
            "wall_seconds": float(bank["wall_seconds"]),
        }
        _append_jsonl(scan_path, record)
        scanned.append(record)
        if record["accepted"]:
            accepted.append(record)

    accepted = accepted[:size]
    selected_ids = sorted(record["item_id"] for record in accepted)
    record_by_id = {record["item_id"]: record for record in accepted}
    panel_items = []
    for index, item_id in enumerate(selected_ids):
        item = dict(by_id[item_id])
        item["panel_index"] = index
        panel_items.append(item)
    write_jsonl(panel_out, panel_items)

    banks = [
        {
            "panel_index": index,
            "item_id": item_id,
            "scan_rank": int(record_by_id[item_id]["scan_rank"]),
            "prompt_tokens": int(record_by_id[item_id]["prompt_tokens"]),
            "candidates": record_by_id[item_id]["candidates"],
        }
        for index, item_id in enumerate(selected_ids)
    ]
    artifact: dict[str, object] = {
        "schema_version": 2,
        "protocol_sha256": file_sha256(protocol_path),
        "panel_file_sha256": file_sha256(panel_out),
        "generator": dict(protocol["generator"]),
        "temperature": GENERATION_TEMPERATURE,
        "top_p": 0.95,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_rule": "GENERATION_SEED_BASE + 1000*scan_rank + slot",
        "seed_base": GENERATION_SEED_BASE,
        "screen": {"min_spread": min_spread, "selection_seed": selection_seed},
        "generation_cost": {
            "wall_seconds": sum(float(r["wall_seconds"]) for r in scanned),
            "input_tokens": sum(int(r["input_tokens"]) for r in scanned),
            "output_tokens": sum(int(r["output_tokens"]) for r in scanned),
        },
        "banks": banks,
    }
    artifact["content_sha256"] = object_sha256(artifact)
    write_json(banks_out, artifact)

    manifest = {
        "schema_version": 2,
        "source_sha256": file_sha256(source_json),
        "selection_seed": selection_seed,
        "size": size,
        "screen": {
            "rule": "admit a question when max-min severity-utility spread of "
            "its frozen candidate bank is at least min_spread; identical rule "
            "for development and held-out panels; gold answers are used only "
            "after generation, to score the frozen bank",
            "min_spread": min_spread,
            "revision": "protocol v2",
        },
        "excluded_item_ids": sorted(excluded),
        "scanned": len(scanned),
        "scanned_item_ids": [record["item_id"] for record in scanned],
        "accepted_flags": [bool(record["accepted"]) for record in scanned],
        "item_ids": selected_ids,
        "acceptance_rate": len(accepted) / len(scanned),
        "panel_sha256": object_sha256(panel_items),
        "panel_file_sha256": file_sha256(panel_out),
        "banks_file_sha256": file_sha256(banks_out),
        "scan_file_sha256": file_sha256(scan_path),
    }
    write_json(manifest_out, manifest)
    return manifest
