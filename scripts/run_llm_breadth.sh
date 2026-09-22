#!/usr/bin/env bash
# Breadth simulations (CPU only, closed grid): high-stakes panel for three
# generators and the cautious review policy for the primary generator,
# each with a lambda=.40 floor cell that includes learned occupancy.
# Requires the GPU-produced kernel artifacts in frozen/ (sbatch scripts
# sbatch_tis_llm_hs1..3.bash and sbatch_tis_llm_pol1.bash).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${TIS_PYTHON_BIN:-python3}"
PROCESSES="${TIS_SIM_PROCESSES:-32}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_one() {
  local config="$1" kernels="$2"
  if [ ! -f "$kernels" ]; then
    echo "== waiting on GPU artifact $kernels; skipping $config"
    return
  fi
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

run_one configs/llm_study_hs.json frozen/hs_qwen_kernels.json
run_one configs/floor_hs_l40/qwen4b.json frozen/hs_qwen_kernels.json
run_one configs/llm_study_cautious_qwen4b.json frozen/cautious_qwen_kernels.json
run_one configs/floor_cautious_qwen4b_l40.json frozen/cautious_qwen_kernels.json
run_one configs/llm_study_hs_robustness.json frozen/hs_robustness_kernels.json
run_one configs/floor_hs_l40/phi4mini.json frozen/hs_robustness_kernels.json
run_one configs/llm_study_hs_scale2.json frozen/hs_scale2_kernels.json
run_one configs/floor_hs_l40/qwen32b.json frozen/hs_scale2_kernels.json
echo "== breadth simulations complete"
