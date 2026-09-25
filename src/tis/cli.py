from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import verify_result_directory
from .runner import run_configuration
from .validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail-Influence Sampling reproducibility artifact")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="run analytic and finite-difference checks")
    run_parser = subparsers.add_parser("run", help="run a deterministic experiment configuration")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--overwrite", action="store_true")
    verify_parser = subparsers.add_parser(
        "verify-results", help="verify hashes, budgets, row counts, and allocation floors"
    )
    verify_parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        print(json.dumps(run_validation(), indent=2, sort_keys=True))
    elif args.command == "run":
        output = run_configuration(args.config, overwrite=args.overwrite)
        print(output)
    elif args.command == "verify-results":
        print(
            json.dumps(
                verify_result_directory(args.directory), indent=2, sort_keys=True
            )
        )


if __name__ == "__main__":
    main()
