#!/usr/bin/env bash
# Utility sensitivity study (CPU only, closed grid, primary generator).
#
# Reruns the full panel of fixed-score methods under the asymmetric
# deployment utility (confident_error: u = (1+c)/2 when correct,
# (1-c)^2/2 when wrong) on the same frozen kernels and seeds, plus one
# lambda=.40 floor cell for the fair floor-selected comparison. Declared
# post hoc as an estimand sensitivity study.
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
  "$PYTHON" -m tis.llm_study.cli run \
    --config "$config" --kernels "$kernels" --output "$outdir" \
    --processes "$PROCESSES" "${overwrite[@]}"
  "$PYTHON" -m tis.llm_study.cli verify-results --directory "$outdir"
}

run_one configs/llm_study_confident_qwen4b.json frozen/qwen_kernels.json
run_one configs/floor_confident_qwen4b_l40.json frozen/qwen_kernels.json
echo "== utility sensitivity complete"
