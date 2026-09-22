#!/usr/bin/env bash
# Anchored-TIS evaluation (CPU only), declared post hoc after the closed-grid
# results showed learned occupancy dominating learned tail influence on the
# language-model suite. The anchored score is the fixed equal-weight ensemble
# of the pilot influence and pilot occupancy designs (mu = 1/2, declared
# once); the prespecified exploration floor lambda_N = N^{-1/4} is unchanged.
# Panels: uniform, oracle, TIS, learned occupancy, anchored TIS — six LLM
# generators (closed grid) plus the controlled/inventory replacement suite
# and the public suite.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${TIS_PYTHON_BIN:-python3}"
PROCESSES="${TIS_SIM_PROCESSES:-16}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_llm() {
  local config="$1" kernels="$2"
  local outdir
  outdir="$("$PYTHON" -c "import json; print(json.load(open('$config'))['output_directory'])")"
  if [ -f "$outdir/manifest.json" ]; then
    echo "== skip $config"
    return
  fi
  echo "== running $config ($kernels)"
  local overwrite=()
  [ -d "$outdir" ] && overwrite=(--overwrite)
  "$PYTHON" -m tis.llm_study.cli run \
    --config "$config" --kernels "$kernels" --output "$outdir" \
    --processes "$PROCESSES" "${overwrite[@]}"
  "$PYTHON" -m tis.llm_study.cli verify-results --directory "$outdir"
}

run_llm configs/llm_study_anchored.json frozen/qwen_kernels.json
run_llm configs/llm_study_anchored_scale3.json frozen/scale3_kernels.json
run_llm configs/llm_study_anchored_robustness.json frozen/robustness_kernels.json
run_llm configs/llm_study_anchored_robustness2.json frozen/robustness2_kernels.json
run_llm configs/llm_study_anchored_scale1.json frozen/scale1_kernels.json
run_llm configs/llm_study_anchored_scale2.json frozen/scale2_kernels.json

for config in configs/replacement_anchored.json configs/public_anchored.json; do
  outdir="$("$PYTHON" -c "import json; print(json.load(open('$config'))['output_directory'])")"
  if [ -f "$outdir/manifest.json" ]; then
    echo "== skip $config"
  else
    echo "== running $config"
    overwrite=()
    [ -d "$outdir" ] && overwrite=(--overwrite)
    "$PYTHON" -m tis.cli run --config "$config" "${overwrite[@]}"
  fi
done
echo "== anchored evaluation complete"
