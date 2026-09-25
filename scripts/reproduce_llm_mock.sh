#!/usr/bin/env bash
set -euo pipefail

python -m tis.llm_study.cli write-ci-mock \
  --output build/llm_mock_kernels.json --questions 2
python -m tis.llm_study.cli run \
  --config configs/llm_study_mock_quick.json \
  --kernels build/llm_mock_kernels.json --overwrite
python -m tis.llm_study.cli verify-results \
  --directory results/llm_mock_quick
