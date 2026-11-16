#!/usr/bin/env bash
# Study 2 of the live review loop (Appendix H) on one machine with GPUS GPUs of one type.
#   bash scripts/run_live_loop_gpu.sh setup     # once: uv, Python 3.12, vLLM, pinned model snapshot
#   bash scripts/run_rare_live_gpu.sh GPUS      # screening, retain, design runs with in-wave references, analysis
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" OPENBLAS_NUM_THREADS=1 HF_HOME="${HF_HOME:-$PWD/.hf}"
export LIVE_LOOP_CONFIG=configs/rebuttal_rare_live.json
PY=.venv-gpu/bin/python
G="${1:-4}"
OUTDIR=results/rebuttal_rare_live
mkdir -p logs "$OUTDIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader > "$OUTDIR/gpus.txt"
if [ ! -f "$OUTDIR/screening.json" ]; then
  for k in $(seq 0 $((G - 1))); do
    CUDA_VISIBLE_DEVICES=$k "$PY" scripts/rebuttal_live_loop.py screen --shard "$k/$G" > "logs/rare_screen_${k}of$G.log" 2>&1 &
  done
  wait
  "$PY" scripts/rebuttal_live_loop.py retain > logs/rare_retain.log 2>&1
fi
for k in $(seq 0 $((G - 1))); do
  CUDA_VISIBLE_DEVICES=$k "$PY" scripts/rebuttal_live_loop.py run --shard "$k/$G" > "logs/rare_run_${k}of$G.log" 2>&1 &
done
wait
"$PY" scripts/rebuttal_live_loop.py analyze > logs/rare_analyze.log 2>&1
echo "analyze exit $?" >> logs/rare_analyze.log
