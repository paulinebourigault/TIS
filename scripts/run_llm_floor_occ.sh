#!/usr/bin/env bash
# Learned-occupancy floor cells (closed grid, H=6, alpha=.1, b=400) at
# lambda .25 and .40 for all six generators, so the per-method-tuned floor
# comparison (each learned method at its cross-validated floor) is fair.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${TIS_PYTHON_BIN:-python3}"
PROCESSES="${TIS_SIM_PROCESSES:-12}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
declare -A KERNELS=(
  [qwen4b]=frozen/qwen_kernels.json
  [phi4mini]=frozen/robustness_kernels.json
  [granite8b]=frozen/robustness2_kernels.json
  [mistral24b]=frozen/scale1_kernels.json
  [qwen32b]=frozen/scale2_kernels.json
  [glm32b]=frozen/scale3_kernels.json
)
for tag in l40 l25; do
  for model in qwen4b phi4mini granite8b mistral24b qwen32b glm32b; do
    config="configs/floor_occ_closed/${model}_${tag}.json"
    outdir="results/llm_floor_occ_closed_${tag}/${model}"
    if [ -f "$outdir/manifest.json" ]; then echo "== skip $config"; continue; fi
    echo "== running $config"
    overwrite=()
    [ -d "$outdir" ] && overwrite=(--overwrite)
    "$PYTHON" -m tis.llm_study.cli run \
      --config "$config" --kernels "${KERNELS[$model]}" --output "$outdir" \
      --processes "$PROCESSES" "${overwrite[@]}"
    "$PYTHON" -m tis.llm_study.cli verify-results --directory "$outdir"
  done
done
echo "== occupancy floor cells complete"
