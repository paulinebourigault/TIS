from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..llm_study.common import read_json
from . import WORKFLOWS
from .calibrate import audit_generations, calibrate_kernels
from .candidates import build_candidate_banks, build_screened_panel
from .panel import DEV_SEED, DEV_SIZE, HELDOUT_SEED, HELDOUT_SIZE, download_source, freeze_panel
from .runner import run_finqa_study


def main() -> None:
    parser = argparse.ArgumentParser(description="FinQA terminal-risk study")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-source")
    fetch.add_argument("--output", type=Path, required=True)

    panels = commands.add_parser("freeze-panels")
    panels.add_argument("--source", type=Path, required=True)
    panels.add_argument("--output-dir", type=Path, required=True)

    screened = commands.add_parser(
        "build-screened-panel",
        help="protocol v2: scan the seeded order, admit questions whose frozen "
        "bank has severity-utility spread >= the declared threshold",
    )
    screened.add_argument("--protocol", type=Path, required=True)
    screened.add_argument("--source", type=Path, required=True)
    screened.add_argument("--output-dir", type=Path, required=True)
    screened.add_argument("--phase", choices=["dev", "heldout"], required=True)
    screened.add_argument(
        "--exclude-manifest", type=Path, default=None,
        help="dev scan manifest whose scanned_item_ids are excluded "
        "(required for --phase heldout)",
    )
    screened.add_argument(
        "--size", type=int, default=None,
        help="panel quota override (default: the phase's declared size); "
        "recorded in the manifest, seeded order unchanged",
    )
    screened.add_argument("--local-files-only", action="store_true")

    cand = commands.add_parser("build-candidates")
    cand.add_argument("--protocol", type=Path, required=True)
    cand.add_argument("--panel-jsonl", type=Path, required=True)
    cand.add_argument("--output", type=Path, required=True)
    cand.add_argument("--local-files-only", action="store_true")

    calib = commands.add_parser("calibrate")
    calib.add_argument("--protocol", type=Path, required=True)
    calib.add_argument("--panel-jsonl", type=Path, required=True)
    calib.add_argument("--banks", type=Path, required=True)
    calib.add_argument("--output", type=Path, required=True)
    calib.add_argument("--workflow", choices=list(WORKFLOWS), required=True)
    calib.add_argument("--local-files-only", action="store_true")

    audit = commands.add_parser("audit-generations")
    audit.add_argument("--artifact", type=Path, required=True)
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--panel-jsonl", type=Path, required=True)
    audit.add_argument("--banks", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--samples", type=int, default=2048)
    audit.add_argument("--batch-size", type=int, default=128)
    audit.add_argument("--local-files-only", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--kernels", type=Path, required=True)
    run.add_argument("--banks", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "fetch-source":
        print(json.dumps(download_source(args.output), indent=2))
    elif args.command == "freeze-panels":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dev = freeze_panel(
            args.source,
            args.output_dir / "finqa_panel_dev.jsonl",
            args.output_dir / "finqa_panel_dev_manifest.json",
            DEV_SIZE, DEV_SEED,
        )
        heldout = freeze_panel(
            args.source,
            args.output_dir / "finqa_panel_heldout.jsonl",
            args.output_dir / "finqa_panel_heldout_manifest.json",
            HELDOUT_SIZE, HELDOUT_SEED,
            exclude_item_ids=dev["item_ids"],
        )
        overlap = set(dev["item_ids"]) & set(heldout["item_ids"])
        assert not overlap
        print(json.dumps({"dev": dev["panel_sha256"], "heldout": heldout["panel_sha256"],
                          "disjoint": True}, indent=2))
    elif args.command == "build-screened-panel":
        protocol = read_json(args.protocol)
        min_spread = float(protocol["panel_screen"]["min_spread"])
        if args.phase == "dev":
            size, seed, tag = DEV_SIZE, DEV_SEED, "dev2"
            exclude: list[str] = []
        else:
            if args.exclude_manifest is None:
                parser.error("--phase heldout requires --exclude-manifest")
            dev_manifest = read_json(args.exclude_manifest)
            exclude = [str(i) for i in dev_manifest["scanned_item_ids"]]
            size, seed, tag = HELDOUT_SIZE, HELDOUT_SEED, "heldout2"
        if args.size is not None:
            size = int(args.size)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_screened_panel(
            args.protocol,
            args.source,
            args.output_dir / f"finqa_panel_{tag}.jsonl",
            args.output_dir / f"finqa_panel_{tag}_manifest.json",
            args.output_dir / f"finqa_banks_{tag}.json",
            args.output_dir / f"finqa_scan_{tag}.jsonl",
            size,
            seed,
            min_spread,
            exclude_item_ids=exclude,
            local_files_only=args.local_files_only,
        )
        print(json.dumps({"phase": args.phase, "scanned": manifest["scanned"],
                          "accepted": len(manifest["item_ids"]),
                          "acceptance_rate": manifest["acceptance_rate"],
                          "panel_sha256": manifest["panel_sha256"]}, indent=2))
    elif args.command == "build-candidates":
        artifact = build_candidate_banks(
            args.protocol, args.panel_jsonl, args.output,
            local_files_only=args.local_files_only,
        )
        valid = sum(
            1 for bank in artifact["banks"] for c in bank["candidates"] if c["valid"]
        )
        total = sum(len(bank["candidates"]) for bank in artifact["banks"])
        print(json.dumps({"output": str(args.output), "valid_candidates": valid,
                          "total_candidates": total,
                          "cost": artifact["generation_cost"]}, indent=2))
    elif args.command == "calibrate":
        artifact = calibrate_kernels(
            args.protocol, args.panel_jsonl, args.banks, args.output,
            workflow=args.workflow, local_files_only=args.local_files_only,
        )
        print(json.dumps({"output": str(args.output), "workflow": args.workflow,
                          "questions": len(artifact["questions"]),
                          "cost": artifact["calibration_cost"]}, indent=2))
    elif args.command == "audit-generations":
        report = audit_generations(
            args.artifact, args.protocol, args.panel_jsonl, args.banks, args.output,
            samples=args.samples, batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        print(json.dumps({"passed": report["passed"],
                          "audited_cells": report["audited_cells"],
                          "smallest_p_value": report["smallest_p_value"]}, indent=2))
    elif args.command == "run":
        print(run_finqa_study(args.config, args.kernels, args.banks,
                              args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
