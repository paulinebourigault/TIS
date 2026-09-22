#!/usr/bin/env bash
# Complete-rollout baseline runs (CPU only, closed grid).
#
# For each generator: uniform and TIS anchors (reproducing the closed-grid
# main panel exactly through the coupled streams) plus the complete-rollout
# empirical-CVaR estimator, which draws floor(N/H) independent full
# trajectories through the same frozen conditional laws at matched query
# cost. Run after (or alongside) run_llm_closedgrid.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${TIS_PYTHON_BIN:-python3}"
PROCESSES="${TIS_SIM_PROCESSES:-32}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

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
  [ -d "$outdir" ] && overwrite=(--overwrite)
  # --output pins the destination: for configs nested below configs/, the
  # runner would otherwise resolve the relative output_directory against the
  # config's grandparent and write under configs/results/.
  "$PYTHON" -m tis.llm_study.cli run \
    --config "$config" --kernels "$kernels" --output "$outdir" \
    --processes "$PROCESSES" "${overwrite[@]}"
  "$PYTHON" -m tis.llm_study.cli verify-results --directory "$outdir"
}

run_one configs/llm_study_rollout.json frozen/qwen_kernels.json
run_one configs/llm_study_rollout_robustness.json frozen/robustness_kernels.json
run_one configs/llm_study_rollout_robustness2.json frozen/robustness2_kernels.json
run_one configs/llm_study_rollout_scale1.json frozen/scale1_kernels.json
run_one configs/llm_study_rollout_scale2.json frozen/scale2_kernels.json
run_one configs/llm_study_rollout_scale3.json frozen/scale3_kernels.json
echo "== rollout baseline complete"
