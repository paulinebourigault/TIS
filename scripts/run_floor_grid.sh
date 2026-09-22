#!/usr/bin/env bash
set -euo pipefail
cd ~/tis
for tag in l15 l40; do
  for pair in qwen_kernels:qwen4b robustness_kernels:phi4mini robustness2_kernels:granite8b scale1_kernels:mistral24b scale2_kernels:qwen32b scale3_kernels:glm32b; do
    k=${pair%%:*}; m=${pair##*:}
    echo "== floor grid $tag $m $(date -Iseconds)"
    OMP_NUM_THREADS=1 .llm-venv/bin/python -m tis.llm_study.cli run --config configs/floor_sensitivity_$tag/$m.json --kernels frozen/$k.json --output results/llm_floor_sensitivity_$tag/$m --overwrite --processes 10
  done
done
echo FLOOR_GRID_DONE
