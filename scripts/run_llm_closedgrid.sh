#!/usr/bin/env bash
# Closed-grid rerun of the language-model study (CPU only).
#
# scripts/grid_representation_check.py showed that the original return grid
# (full-horizon sums plus endpoints) leaves intermediate-layer return-to-go
# supports off the grid, so the stop-loss recursion interpolates at depth
# two and beyond and the categorical target can understate the exact-law
# CVaR in low-tail cells. The closed grid (union of all attainable
# partial-sum supports, config flag "closed_grid": true) makes the
# recursion interpolation-free: the categorical target equals the exact
# CVaR to machine precision. This script reruns the full panel, the
# fair-baseline extension, and the floor grid under the closed grid with
# the same frozen kernels and master seed. The original-grid results stay
# in the repository as the as-executed initial run.
#
# Usage: run_llm_closedgrid.sh <a|b>   (two lanes, run one per invocation)
set -euo pipefail
cd "$(dirname "$0")/.."

LANE="${1:?usage: run_llm_closedgrid.sh <a|b>}"
PYTHON="${TIS_PYTHON_BIN:-python3}"
PROCESSES="${TIS_SIM_PROCESSES:-22}"

# One BLAS thread per worker: the solver's matrices are small, and letting
# OpenBLAS spawn a thread per core in every worker oversubscribes the host
# by orders of magnitude (observed: 48 threads x 44 workers, ~20% useful CPU).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

declare -A KERNELS=(
  [qwen4b]=frozen/qwen_kernels.json
  [phi4mini]=frozen/robustness_kernels.json
  [granite8b]=frozen/robustness2_kernels.json
  [mistral24b]=frozen/scale1_kernels.json
  [qwen32b]=frozen/scale2_kernels.json
  [glm32b]=frozen/scale3_kernels.json
)
declare -A SUFFIX=(
  [qwen4b]=""
  [phi4mini]="_robustness"
  [granite8b]="_robustness2"
  [mistral24b]="_scale1"
  [qwen32b]="_scale2"
  [glm32b]="_scale3"
)
if [ "$LANE" = "a" ]; then
  MODELS=(qwen4b granite8b qwen32b)
else
  MODELS=(phi4mini mistral24b glm32b)
fi

run_one() {
  local config="$1" kernels="$2"
  local outdir
  outdir="$("$PYTHON" -c "import json; print(json.load(open('$config'))['output_directory'])")"
  if [ -f "$outdir/manifest.json" ]; then
    echo "== skip $config: $outdir already complete"
    return
  fi
  echo "== running $config ($kernels) with $PROCESSES processes"
  local overwrite=()
  [ -d "$outdir" ] && overwrite=(--overwrite)  # cells are cached; restart is cheap
  # --output pins the destination: for configs nested below configs/, the
  # runner would otherwise resolve the relative output_directory against the
  # config's grandparent and write under configs/results/.
  "$PYTHON" -m tis.llm_study.cli run \
    --config "$config" --kernels "$kernels" --output "$outdir" \
    --processes "$PROCESSES" "${overwrite[@]}"
  "$PYTHON" -m tis.llm_study.cli verify-results --directory "$outdir"
}

for model in "${MODELS[@]}"; do
  kernels="${KERNELS[$model]}"
  suffix="${SUFFIX[$model]}"
  run_one "configs/llm_study_closed${suffix}.json" "$kernels"
  run_one "configs/llm_study_ext_closed${suffix}.json" "$kernels"
  for floor in floor_sensitivity_closed floor_sensitivity_l15_closed floor_sensitivity_l40_closed; do
    run_one "configs/${floor}/${model}.json" "$kernels"
  done
done
echo "== closed-grid lane $LANE complete"
