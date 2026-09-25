from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .reporting import sha256
from .runner import METHODS


def verify_result_directory(directory: Path) -> dict[str, object]:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

    repository_root = (
        directory / manifest.get("source_root_from_result", "../..")
    ).resolve()
    for relative, expected in manifest.get("source_files", {}).items():
        path = repository_root / relative
        if not path.is_file():
            failures.append(f"missing source file: {relative}")
            continue
        if path.stat().st_size != int(expected["bytes"]):
            failures.append(f"source size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            failures.append(f"source hash mismatch: {relative}")

    config = manifest["configuration"]
    default_methods = tuple(config.get("methods", METHODS))
    method_panels = {
        environment["name"]: set(environment.get("methods", default_methods))
        for environment in config["environments"]
    }
    expected_raw_rows = sum(
        len(method_panels[environment["name"]])
        * int(environment.get("replications", config["replications"]))
        * len(environment.get("budgets_per_group", environment.get("total_budgets", [])))
        for environment in config["environments"]
    )
    raw_by_key: dict[tuple[str, int, int, str], dict[str, int]] = {}
    raw_panels: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    with (directory / "raw.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        raw_count = 0
        for row in reader:
            raw_count += 1
            if float(row["squared_error"]) < 0:
                failures.append("negative squared error")
            total = int(row["total_budget"])
            pilot = int(row["pilot_total"])
            main = int(row["main_total"])
            if pilot + main != total:
                failures.append(
                    f"budget mismatch: {row['environment']} {row['budget_per_group']} "
                    f"rep={row['replication']} method={row['method']}"
                )
            key = (
                row["environment"],
                int(row["total_budget"]),
                int(row["replication"]),
                row["method"],
            )
            raw_by_key[key] = {"main": main, "total": total}
            raw_panels[key[:3]].add(row["method"])
    if raw_count != expected_raw_rows:
        failures.append(
            f"raw row count {raw_count} != expected {expected_raw_rows}"
        )
    for panel, methods in raw_panels.items():
        if methods != method_panels[panel[0]]:
            failures.append(f"method panel mismatch: {panel}")

    groups_by_environment = {
        diagnostic["environment"]: int(diagnostic["groups"])
        for diagnostic in manifest["diagnostics"]
    }
    allocation_sums: dict[tuple[str, int, int, str], int] = defaultdict(int)
    allocation_count = 0
    with (directory / "allocations.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            allocation_count += 1
            key = (
                row["environment"],
                int(
                    row.get("total_budget")
                    or round(
                        float(row["budget_per_group"])
                        * groups_by_environment[row["environment"]]
                    )
                ),
                int(row["replication"]),
                row["method"],
            )
            count = int(row["main_count"])
            if count < int(config["minimum_main_per_group"]):
                failures.append(f"main floor violation: {key}")
            allocation_sums[key] += count
    expected_allocation_rows = sum(
        len(method_panels[environment["name"]])
        * int(environment.get("replications", config["replications"]))
        * len(environment.get("budgets_per_group", environment.get("total_budgets", [])))
        * groups_by_environment[environment["name"]]
        for environment in config["environments"]
    )
    if allocation_count != expected_allocation_rows:
        failures.append(
            f"allocation row count {allocation_count} != expected {expected_allocation_rows}"
        )
    if set(allocation_sums) != set(raw_by_key):
        failures.append("allocation/raw key mismatch")
    for key, total in allocation_sums.items():
        if key in raw_by_key and total != raw_by_key[key]["main"]:
            failures.append(f"allocation total mismatch: {key}")

    report = {
        "directory": str(directory),
        "raw_rows": raw_count,
        "allocation_rows": allocation_count,
        "methods": {name: sorted(methods) for name, methods in method_panels.items()},
        "hash_files": len(manifest["files"]),
        "source_hash_files": len(manifest.get("source_files", {})),
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise RuntimeError(json.dumps(report, indent=2))
    return report
