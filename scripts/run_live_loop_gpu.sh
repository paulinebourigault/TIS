#!/usr/bin/env bash
# Study 1 of the live review loop (Appendix H) on a GPU machine.
#   bash scripts/run_live_loop_gpu.sh setup        # uv, Python 3.12, vLLM, pinned model snapshot
#   bash scripts/run_live_loop_gpu.sh smoke        # format and throughput (no estimator)
#   bash scripts/run_live_loop_gpu.sh full GPUS    # scale rule, screening, reference, designs, analysis
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" OPENBLAS_NUM_THREADS=1 HF_HOME="${HF_HOME:-$PWD/.hf}"
PY=.venv-gpu/bin/python
mkdir -p logs
case "${1:-}" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    uv venv --python 3.12 .venv-gpu
    uv pip install --python "$PY" vllm==0.10.1.1 transformers==4.55.4 matplotlib==3.10.8
    "$PY" -c "import vllm, torch, numpy; print('vllm', vllm.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
    "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554'))"
    ;;
  smoke)
    "$PY" scripts/rebuttal_live_loop.py smoke 2>&1 | tee logs/live_smoke.log
    ;;
  full)
    G="${2:-1}"
    "$PY" scripts/rebuttal_live_loop.py decide "$G" | tee logs/live_decide.log
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/rebuttal_live_loop.py screen 2>&1 | tee logs/live_screen.log
    for k in $(seq 0 $((G - 1))); do
      CUDA_VISIBLE_DEVICES=$k "$PY" scripts/rebuttal_live_loop.py reference --shard "$k/$G" > "logs/live_reference_$k.log" 2>&1 &
    done
    wait
    for k in $(seq 0 $((G - 1))); do
      CUDA_VISIBLE_DEVICES=$k "$PY" scripts/rebuttal_live_loop.py run --shard "$k/$G" > "logs/live_run_$k.log" 2>&1 &
    done
    wait
    "$PY" scripts/rebuttal_live_loop.py analyze 2>&1 | tee logs/live_analyze.log
    ;;
  *)
    sed -n '2,5p' "$0"; exit 1
    ;;
esac
