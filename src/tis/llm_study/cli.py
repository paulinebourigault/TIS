from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import verify_llm_result_directory
from .audit import audit_generations
from .calibrate import MODEL_KEYS, calibrate_kernels, validate_kernel_artifact
from .common import read_json
from .mock import write_mock_kernel_artifact
from .panel import export_pinned_mmlu_pro, freeze_panel
from .runner import run_llm_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate structured language-model workflow study"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-source")
    export.add_argument("--output-jsonl", type=Path, required=True)
    export.add_argument("--protocol", type=Path, required=True)
    panel = commands.add_parser("freeze-panel")
    panel.add_argument("--source-jsonl", type=Path, required=True)
    panel.add_argument("--panel-jsonl", type=Path, required=True)
    panel.add_argument("--manifest", type=Path, required=True)
    panel.add_argument("--protocol", type=Path, required=True)
    panel.add_argument(
        "--panel-section",
        default="panel",
        help="Protocol section holding this panel's subject counts and seed",
    )
    panel.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Existing panel manifest whose item_ids are excluded from selection",
    )
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--protocol", type=Path, required=True)
    calibrate.add_argument("--panel-jsonl", type=Path, required=True)
    calibrate.add_argument("--panel-manifest", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--local-files-only", action="store_true")
    calibrate.add_argument(
        "--model-key",
        choices=list(MODEL_KEYS),
        default="model",
        help="Protocol model section to calibrate",
    )
    calibrate.add_argument(
        "--policy",
        default="default",
        help="Review policy variant mapping confidence bands to actions",
    )
    calibrate.add_argument(
        "--panel-section",
        default="panel",
        help="Protocol section describing the calibrated panel",
    )
    validate = commands.add_parser("validate-kernels")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--protocol", type=Path)
    validate.add_argument("--panel-jsonl", type=Path)
    audit = commands.add_parser("audit-generations")
    audit.add_argument("--artifact", type=Path, required=True)
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--panel-jsonl", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--questions", type=int, default=3)
    audit.add_argument("--states", type=str, default="0,17,44,71,90")
    audit.add_argument("--samples", type=int, default=4096)
    audit.add_argument("--batch-size", type=int, default=128)
    audit.add_argument("--local-files-only", action="store_true")
    audit.add_argument(
        "--model-key",
        choices=list(MODEL_KEYS),
        default="model",
        help="Protocol model section the audited artifact was calibrated with",
    )
    mock = commands.add_parser("write-ci-mock")
    mock.add_argument("--output", type=Path, required=True)
    mock.add_argument("--questions", type=int, default=2)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--kernels", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker processes over (question, horizon) cells; output is "
        "identical for every value",
    )
    verify_results = commands.add_parser("verify-results")
    verify_results.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export-source":
        protocol = read_json(args.protocol)
        print(json.dumps(export_pinned_mmlu_pro(
            args.output_jsonl,
            str(protocol["panel"]["source_revision"]),
            str(protocol["panel"]["split"]),
        ), indent=2, sort_keys=True))
    elif args.command == "freeze-panel":
        protocol = read_json(args.protocol)
        section = protocol[args.panel_section]
        exclude_item_ids: list[str] = []
        if args.exclude_manifest is not None:
            exclude_item_ids = [
                str(identifier)
                for identifier in read_json(args.exclude_manifest)["item_ids"]
            ]
        manifest = freeze_panel(
            args.source_jsonl,
            args.panel_jsonl,
            args.manifest,
            str(section["source_revision"]),
            section["subject_counts"],
            int(section.get("selection_seed", protocol["seeds"]["panel_selection"])),
            exclude_item_ids=exclude_item_ids,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "calibrate":
        artifact = calibrate_kernels(
            args.protocol,
            args.panel_jsonl,
            args.panel_manifest,
            args.output,
            local_files_only=args.local_files_only,
            model_key=args.model_key,
            policy=args.policy,
            panel_section=args.panel_section,
        )
        print(json.dumps({
            "output": str(args.output),
            "model_key": args.model_key,
            "content_sha256": artifact["content_sha256"],
            "questions": len(artifact["questions"]),
        }, indent=2, sort_keys=True))
    elif args.command == "validate-kernels":
        print(json.dumps(
            validate_kernel_artifact(args.artifact, args.protocol, args.panel_jsonl),
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "audit-generations":
        report = audit_generations(
            args.artifact,
            args.protocol,
            args.panel_jsonl,
            args.output,
            questions=args.questions,
            state_indices=[int(v) for v in args.states.split(",")],
            samples=args.samples,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
            model_key=args.model_key,
        )
        print(json.dumps({
            "output": str(args.output),
            "passed": report["passed"],
            "audited_cells": report["audited_cells"],
            "smallest_p_value": report["smallest_p_value"],
        }, indent=2, sort_keys=True))
    elif args.command == "write-ci-mock":
        artifact = write_mock_kernel_artifact(args.output, args.questions)
        print(json.dumps({
            "output": str(args.output),
            "evidence_status": artifact["evidence_status"],
            "questions": len(artifact["questions"]),
        }, indent=2, sort_keys=True))
    elif args.command == "run":
        print(run_llm_study(
            args.config,
            args.kernels,
            output_directory=args.output,
            overwrite=args.overwrite,
            processes=args.processes,
        ))
    elif args.command == "verify-results":
        print(json.dumps(
            verify_llm_result_directory(args.directory),
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
