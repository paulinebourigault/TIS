#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m tis validate
python -m tis run --config configs/public_paper.json --overwrite
python -m tis verify-results --directory results/public_paper
