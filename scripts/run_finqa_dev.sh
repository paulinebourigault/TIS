#!/usr/bin/env bash
# FinQA development simulations (CPU): both workflows, all methods.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${TIS_PYTHON_BIN:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for wf in ordinary unitcheck; do
  out="results/finqa_dev_${wf}"
  if [ -f "$out/manifest.json" ]; then echo "== skip $wf"; continue; fi
  echo "== running finqa dev $wf"
  "$PYTHON" -m tis.finqa.cli run \
    --config "configs/finqa_dev_${wf}.json" \
    --kernels "frozen/finqa_kernels_dev2_${wf}.json" \
    --banks frozen/finqa_banks_dev2.json \
    --output "$out" $([ -d "$out" ] && echo --overwrite)
done
echo "== finqa dev simulations complete"
