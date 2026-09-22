"""FinQA source pinning and panel selection.

The source is the FinQA repository at a pinned commit; eligibility keeps
items whose executable gold answer is a finite nonzero float and whose
linearized context fits the declared budget. Selection is the study's
seeded SHA-256 rank rule. The development panel (20 items) and held-out
panel (50 items) are disjoint by construction: held-out selection excludes
every development item id.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

from ..llm_study.common import file_sha256, object_sha256, write_json, write_jsonl

SOURCE_REPO = "czyssrs/FinQA"
SOURCE_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
SOURCE_SPLIT = "dev"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/dataset/dev.json"
)
MAX_CONTEXT_CHARS = 4000
MAX_TEXT_SENTENCES = 12
DEV_SIZE = 20
HELDOUT_SIZE = 50
DEV_SEED = 20260912
HELDOUT_SEED = 20260913


def download_source(output_json: Path) -> dict[str, object]:
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        payload = response.read()
    output_json.write_bytes(payload)
    return {
        "repo": SOURCE_REPO,
        "commit": SOURCE_COMMIT,
        "split": SOURCE_SPLIT,
        "bytes": len(payload),
        "sha256": file_sha256(output_json),
    }


def _linearize_table(table: Sequence[Sequence[str]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def _context(row: Mapping[str, object]) -> str:
    sentences = list(row.get("pre_text") or []) + list(row.get("post_text") or [])
    text = " ".join(str(s) for s in sentences[:MAX_TEXT_SENTENCES])
    table = _linearize_table(row.get("table") or [])
    return f"Report excerpt:\n{text}\n\nTable:\n{table}"


def normalize_items(source_json: Path) -> list[dict[str, object]]:
    rows = json.loads(source_json.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = []
    for row in rows:
        qa = row.get("qa") or {}
        answer = qa.get("exe_ans")
        if not isinstance(answer, (int, float)):
            continue
        answer = float(answer)
        if not (answer == answer) or answer in (float("inf"), float("-inf")):
            continue
        if abs(answer) < 1e-9:
            continue
        context = _context(row)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]
        items.append(
            {
                "item_id": str(row["id"]),
                "question": str(qa["question"]).strip(),
                "context": context,
                "gold_answer": answer,
                "gold_program": str(qa.get("program", "")),
            }
        )
    return items


def _seeded_rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}|{item_id}".encode("utf-8")).hexdigest()


def freeze_panel(
    source_json: Path,
    output_jsonl: Path,
    manifest_path: Path,
    size: int,
    selection_seed: int,
    exclude_item_ids: Sequence[str] = (),
) -> dict[str, object]:
    items = normalize_items(source_json)
    excluded = {str(i) for i in exclude_item_ids}
    eligible = [item for item in items if item["item_id"] not in excluded]
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} eligible items for panel of {size}")
    ordered = sorted(eligible, key=lambda it: _seeded_rank(selection_seed, it["item_id"]))
    selected = sorted(ordered[:size], key=lambda it: it["item_id"])
    for index, item in enumerate(selected):
        item["panel_index"] = index
    write_jsonl(output_jsonl, selected)
    manifest = {
        "schema_version": 1,
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_split": SOURCE_SPLIT,
        "source_sha256": file_sha256(source_json),
        "eligibility": "finite nonzero float exe_ans; context truncated to "
        f"{MAX_TEXT_SENTENCES} sentences and {MAX_CONTEXT_CHARS} characters",
        "selection_seed": selection_seed,
        "size": size,
        "excluded_item_ids": sorted(excluded),
        "item_ids": [item["item_id"] for item in selected],
        "panel_sha256": object_sha256(selected),
        "panel_file_sha256": file_sha256(output_jsonl),
    }
    write_json(manifest_path, manifest)
    return manifest
