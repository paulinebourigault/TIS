from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .common import file_sha256, object_sha256, read_jsonl, write_json, write_jsonl


def export_pinned_mmlu_pro(
    output_jsonl: Path, dataset_revision: str, split: str = "test"
) -> dict[str, object]:
    if len(dataset_revision) != 40:
        raise ValueError("Dataset revision must be a full 40-character commit")
    try:
        import datasets
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Pinned MMLU-Pro export requires requirements-llm-lock.txt"
        ) from exc
    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro", revision=dataset_revision, split=split
    )
    rows = [dict(row) for row in dataset]
    write_jsonl(output_jsonl, rows)
    return {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "revision": dataset_revision,
        "split": split,
        "rows": len(rows),
        "sha256": file_sha256(output_jsonl),
        "datasets_version": datasets.__version__,
    }


def _stable_identifier(row: Mapping[str, object], row_index: int) -> str:
    for key in ("question_id", "id", "item_id"):
        if key in row and str(row[key]).strip():
            return str(row[key])
    return f"source-row-{row_index:06d}"


def _normalize_item(
    row: Mapping[str, object], row_index: int
) -> dict[str, object] | None:
    options = row.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError(f"Item {row_index} has no option sequence")
    options = list(options)
    if len(options) != 10 or any(str(value).strip().upper() == "N/A" for value in options):
        return None
    category = str(row.get("category", row.get("subject", ""))).strip().lower()
    if not category:
        raise ValueError(f"Item {row_index} has no category")
    answer_index = row.get("answer_index")
    if answer_index is None:
        answer = str(row.get("answer", "")).strip().upper()
        if len(answer) != 1 or not "A" <= answer <= "J":
            raise ValueError(f"Item {row_index} has invalid answer")
        answer_index = ord(answer) - ord("A")
    answer_index = int(answer_index)
    if not 0 <= answer_index < 10:
        raise ValueError(f"Item {row_index} has invalid answer index")
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError(f"Item {row_index} has no question")
    return {
        "item_id": _stable_identifier(row, row_index),
        "source_row": row_index,
        "subject": category,
        "question": question,
        "options": [str(option) for option in options],
        "correct_index": answer_index,
    }


def freeze_panel(
    source_jsonl: Path,
    output_jsonl: Path,
    manifest_path: Path,
    source_revision: str,
    subject_counts: Mapping[str, int],
    selection_seed: int,
    exclude_item_ids: Sequence[str] = (),
) -> dict[str, object]:
    if len(source_revision) < 12 or source_revision.lower() in {"main", "latest"}:
        raise ValueError("A pinned source commit/revision is required")
    source_rows = read_jsonl(source_jsonl)
    candidates = [
        _normalize_item(row, index) for index, row in enumerate(source_rows)
    ]
    excluded = {str(identifier) for identifier in exclude_item_ids}
    normalized = [
        item
        for item in candidates
        if item is not None and str(item["item_id"]) not in excluded
    ]
    by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in normalized:
        by_subject[str(item["subject"])].append(item)
    selected: list[dict[str, object]] = []
    for subject, count in subject_counts.items():
        subject = subject.lower()
        candidates = by_subject.get(subject, [])
        if len(candidates) < int(count):
            raise ValueError(
                f"Subject {subject!r} has {len(candidates)} items, needs {count}"
            )
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{selection_seed}|{item['item_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[: int(count)])
    if len(selected) != 50:
        raise ValueError(f"Subject counts must select exactly 50 items, got {len(selected)}")
    selected.sort(key=lambda item: (str(item["subject"]), str(item["item_id"])))
    for panel_index, item in enumerate(selected):
        item["panel_index"] = panel_index
    if len({str(item["item_id"]) for item in selected}) != len(selected):
        raise ValueError("Selected item identifiers are not unique")
    write_jsonl(output_jsonl, selected)
    manifest = {
        "schema_version": 1,
        "dataset": "TIGER-Lab/MMLU-Pro",
        "source_revision": source_revision,
        "source_file": source_jsonl.name,
        "source_sha256": file_sha256(source_jsonl),
        "source_rows": len(source_rows),
        "eligible_ten_option_rows": len(normalized),
        "eligibility_rule": "exactly ten options and no N/A placeholder",
        "selection_seed": selection_seed,
        "excluded_item_ids": sorted(excluded),
        "ordering_rule": "subject then item_id, both ascending after seeded hash selection within subject",
        "subject_counts": dict(sorted(Counter(str(item["subject"]) for item in selected).items())),
        "item_ids": [str(item["item_id"]) for item in selected],
        "panel_sha256": object_sha256(selected),
        "panel_file_sha256": file_sha256(output_jsonl),
        "retained_columns": [
            "panel_index", "item_id", "source_row", "subject", "question",
            "options", "correct_index"
        ],
    }
    write_json(manifest_path, manifest)
    return manifest
