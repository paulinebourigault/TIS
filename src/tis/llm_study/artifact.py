from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from ..reporting import sha256


def verify_llm_result_directory(directory: Path) -> dict[str, object]:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != int(expected["bytes"]):
            failures.append(f"size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            failures.append(f"hash mismatch: {relative}")
    methods = set(str(method) for method in config["methods"])
    questions = int(manifest["questions"])
    replications = int(config["replications"])
    expected_panels = (
        len(config["horizons"])
        * len(config["alphas"])
        * len(config["budgets_per_shared_group"])
        * questions
        * replications
    )
    expected_raw = expected_panels * len(methods)
    panels: dict[tuple[str, ...], set[str]] = defaultdict(set)
    raw_count = 0
    with (directory / "raw.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw_count += 1
            if row["method"] == "complete_rollout":
                # Full trajectories consume horizon queries per run; the
                # remainder total_budget mod horizon is charged but unusable.
                usable = int(row["total_budget"]) - (
                    int(row["total_budget"]) % int(row["horizon"])
                )
                if int(row["pilot_total"]) != 0 or int(row["main_total"]) != usable:
                    failures.append("rollout budget mismatch")
            elif int(row["pilot_total"]) + int(row["main_total"]) != int(row["total_budget"]):
                failures.append("pilot/main budget mismatch")
            if int(row["logical_queries"]) != int(row["total_budget"]):
                failures.append("logical-query budget mismatch")
            horizon = int(row["horizon"])
            if row["method"] == "untied_tis":
                if row["group_mode"] != "untied_layer_state":
                    failures.append("untied TIS has wrong group mode")
                if int(row["groups"]) != 1 + 90 * (horizon - 1):
                    failures.append("untied TIS group count mismatch")
            else:
                if row["group_mode"] != "shared_state_action" or int(row["groups"]) != 91:
                    failures.append("shared LLM method group structure mismatch")
            panels[
                (
                    row["horizon"], row["alpha"], row["budget_per_shared_group"],
                    row["panel_index"], row["replication"]
                )
            ].add(row["method"])
    if raw_count != expected_raw:
        failures.append(f"raw row count {raw_count} != {expected_raw}")
    if len(panels) != expected_panels:
        failures.append(f"panel count {len(panels)} != {expected_panels}")
    for key, observed in panels.items():
        if observed != methods:
            failures.append(f"method panel mismatch: {key}")
            break
    expected_panel_replications = (
        len(config["horizons"])
        * len(config["alphas"])
        * len(config["budgets_per_shared_group"])
        * replications
        * len(methods)
    )
    with (directory / "panel_replications.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        panel_replication_count = sum(1 for _ in csv.DictReader(stream))
    if panel_replication_count != expected_panel_replications:
        failures.append(
            f"panel replication row count {panel_replication_count} != "
            f"{expected_panel_replications}"
        )
    if manifest["result_evidence_status"] == "paper_eligible_complete":
        if manifest.get("kernel_evidence_status") != "paper_eligible":
            failures.append("complete paper result uses a non-paper kernel artifact")
        if questions != 50 or replications != 300:
            failures.append("complete paper result has wrong panel or replication count")
    report = {
        "directory": str(directory),
        "result_evidence_status": manifest["result_evidence_status"],
        "raw_rows": raw_count,
        "panels": len(panels),
        "methods": sorted(methods),
        "hash_files": len(manifest["files"]),
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise RuntimeError(json.dumps(report, indent=2))
    return report

