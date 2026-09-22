#!/usr/bin/env bash
# Fair-baseline extension of the language-model study (CPU only).
#
# Reuses the six frozen, audited kernel artifacts; no GPU, model download,
# or network access is required. For every generator it runs the extension
# panel (uniform, oracle_tail, tis anchors plus learned_mean,
# learned_occupancy, tis_no_cov, oracle_no_cov) over the full prespecified
# grid with the primary study's master seed. The anchor methods reproduce
# the primary panel's coupled-stream errors exactly, which doubles as a
# cross-run consistency check.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${TIS_PYTHON_BIN:-python3}"
PROCESSES="${TIS_SIM_PROCESSES:-32}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_extension() {
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

run_extension configs/llm_study_ext.json frozen/qwen_kernels.json
run_extension configs/llm_study_ext_robustness.json frozen/robustness_kernels.json
run_extension configs/llm_study_ext_robustness2.json frozen/robustness2_kernels.json
run_extension configs/llm_study_ext_scale1.json frozen/scale1_kernels.json
run_extension configs/llm_study_ext_scale2.json frozen/scale2_kernels.json
run_extension configs/llm_study_ext_scale3.json frozen/scale3_kernels.json
echo "== extension complete"
