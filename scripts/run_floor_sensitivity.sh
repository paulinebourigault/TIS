#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${TIS_PYTHON_BIN:-python3}"
for pair in scale3_kernels:glm32b scale2_kernels:qwen32b qwen_kernels:qwen4b robustness_kernels:phi4mini robustness2_kernels:granite8b scale1_kernels:mistral24b; do
  k=${pair%%:*}; m=${pair##*:}
  echo "== floor sensitivity: $m $(date -Iseconds)"
  OMP_NUM_THREADS=1 "$PYTHON" -m tis.llm_study.cli run --config configs/floor_sensitivity/$m.json --kernels frozen/$k.json --output results/llm_floor_sensitivity/$m --overwrite --processes 12
done
echo FLOOR_SENSITIVITY_DONE
