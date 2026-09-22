#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

mkdir -p build external frozen results runlogs
RUN_LOG="runlogs/llm_h100_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$RUN_LOG") 2>&1

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${TIS_CPU_THREADS:-16}"
export MKL_NUM_THREADS="${TIS_CPU_THREADS:-16}"

PYTHON_COMMAND="${TIS_PYTHON_BIN:-python3}"
VIRTUAL_ENVIRONMENT=".llm-venv"
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [[ ! -x "$VIRTUAL_ENVIRONMENT/bin/python" ]]; then
  if ! "$PYTHON_COMMAND" -m venv "$VIRTUAL_ENVIRONMENT" 2>/dev/null; then
    echo "python3 -m venv is unavailable; falling back to uv"
    rm -rf "$VIRTUAL_ENVIRONMENT"
    "$UV_BIN" venv --python "$PYTHON_COMMAND" "$VIRTUAL_ENVIRONMENT"
  fi
fi
PYTHON="$VIRTUAL_ENVIRONMENT/bin/python"

echo "== Installing frozen environments =="
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m pip install -r requirements-lock.txt -r requirements-llm-lock.txt
  "$PYTHON" -m pip install --no-deps -e .
else
  "$UV_BIN" pip install --python "$PYTHON" -r requirements-lock.txt -r requirements-llm-lock.txt
  "$UV_BIN" pip install --python "$PYTHON" --no-deps -e .
fi

echo "== Verifying H100/CUDA visibility =="
nvidia-smi
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
device = torch.cuda.get_device_name(0)
if "H100" not in device.upper():
    raise SystemExit(f"Expected an H100, found {device!r}")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": device})
PY

echo "== Running repository and CI-only workflow checks =="
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m tis validate
"$PYTHON" -m tis.llm_study.cli write-ci-mock \
  --output build/llm_mock_kernels.json --questions 2
"$PYTHON" -m tis.llm_study.cli run \
  --config configs/llm_study_mock_quick.json \
  --kernels build/llm_mock_kernels.json --overwrite
"$PYTHON" -m tis.llm_study.cli verify-results \
  --directory results/llm_mock_quick

SOURCE_JSONL="external/mmlu_pro_test_5b0b058.jsonl"
PANEL_JSONL="frozen/llm_panel.jsonl"
PANEL_MANIFEST="frozen/llm_panel_manifest.json"

if [[ ! -f "$SOURCE_JSONL" ]]; then
  echo "== Exporting the pinned MMLU-Pro test split =="
  "$PYTHON" -m tis.llm_study.cli export-source \
    --output-jsonl "$SOURCE_JSONL" \
    --protocol configs/llm_study_protocol.json
else
  echo "== Reusing existing pinned MMLU-Pro export =="
fi

echo "== Freezing the deterministic 50-item panel =="
"$PYTHON" -m tis.llm_study.cli freeze-panel \
  --source-jsonl "$SOURCE_JSONL" \
  --panel-jsonl "$PANEL_JSONL" \
  --manifest "$PANEL_MANIFEST" \
  --protocol configs/llm_study_protocol.json

run_model_chain() {
  local MODEL_KEY="$1"
  local KERNEL_ARTIFACT="$2"
  local AUDIT_REPORT="$3"
  local RUN_CONFIG="$4"
  local RESULT_DIRECTORY="$5"
  local AUDIT_BATCH="${6:-128}"

  if [[ -f "$KERNEL_ARTIFACT" ]]; then
    echo "== Validating existing $MODEL_KEY kernel artifact =="
    if ! "$PYTHON" -m tis.llm_study.cli validate-kernels \
      --artifact "$KERNEL_ARTIFACT" \
      --protocol configs/llm_study_protocol.json \
      --panel-jsonl "$PANEL_JSONL"; then
      local INVALID_KERNEL="${KERNEL_ARTIFACT}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$KERNEL_ARTIFACT" "$INVALID_KERNEL"
      echo "Moved invalid kernel artifact to $INVALID_KERNEL"
    fi
  fi

  if [[ ! -f "$KERNEL_ARTIFACT" ]]; then
    echo "== Enumerating 4,550 real $MODEL_KEY prompt kernels on the H100 =="
    "$PYTHON" -m tis.llm_study.cli calibrate \
      --protocol configs/llm_study_protocol.json \
      --panel-jsonl "$PANEL_JSONL" \
      --panel-manifest "$PANEL_MANIFEST" \
      --model-key "$MODEL_KEY" \
      --output "$KERNEL_ARTIFACT"
  fi

  echo "== Enforcing real-kernel provenance and probability checks =="
  "$PYTHON" -m tis.llm_study.cli validate-kernels \
    --artifact "$KERNEL_ARTIFACT" \
    --protocol configs/llm_study_protocol.json \
    --panel-jsonl "$PANEL_JSONL"

  if [[ -f "$AUDIT_REPORT" ]] && "$PYTHON" - "$AUDIT_REPORT" "$KERNEL_ARTIFACT" <<'PY'
import json, sys, hashlib
report = json.load(open(sys.argv[1]))
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
ok = report.get("passed") is True and report.get("kernel_artifact_sha256") == digest
raise SystemExit(0 if ok else 1)
PY
  then
    echo "== Reusing passing generation audit for $KERNEL_ARTIFACT =="
  else
    echo "== Auditing $MODEL_KEY constrained generations against enumerated kernels =="
    "$PYTHON" -m tis.llm_study.cli audit-generations \
      --artifact "$KERNEL_ARTIFACT" \
      --protocol configs/llm_study_protocol.json \
      --panel-jsonl "$PANEL_JSONL" \
      --model-key "$MODEL_KEY" \
      --batch-size "$AUDIT_BATCH" \
      --output "$AUDIT_REPORT"
  fi

  if [[ -d "$RESULT_DIRECTORY" ]]; then
    echo "== Checking existing complete result directory $RESULT_DIRECTORY =="
    if "$PYTHON" -m tis.llm_study.cli verify-results \
      --directory "$RESULT_DIRECTORY"; then
      echo "Existing complete results are valid; simulation will not be repeated."
    else
      local INCOMPLETE_RESULT="${RESULT_DIRECTORY}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$RESULT_DIRECTORY" "$INCOMPLETE_RESULT"
      echo "Moved incomplete results to $INCOMPLETE_RESULT"
    fi
  fi

  if [[ ! -d "$RESULT_DIRECTORY" ]]; then
    echo "== Running the complete 50-question confirmatory study ($MODEL_KEY) =="
    local SIMULATION_PROCESSES="${TIS_SIM_PROCESSES:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
    echo "Using $SIMULATION_PROCESSES deterministic worker processes"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -m tis.llm_study.cli run \
      --config "$RUN_CONFIG" \
      --kernels "$KERNEL_ARTIFACT" \
      --processes "$SIMULATION_PROCESSES"
  fi

  echo "== Final evidence verification for $RESULT_DIRECTORY =="
  "$PYTHON" -m tis.llm_study.cli verify-results \
    --directory "$RESULT_DIRECTORY"
}

run_model_chain model \
  frozen/qwen_kernels.json \
  frozen/generation_audit.json \
  configs/llm_study_paper.json \
  results/llm_study_paper

if [[ "${TIS_SKIP_ROBUSTNESS:-0}" == "1" ]]; then
  echo "== Skipping the prespecified robustness chains (TIS_SKIP_ROBUSTNESS=1) =="
else
  echo "== Prespecified robustness checkpoint 1: microsoft/Phi-4-mini-instruct =="
  run_model_chain robustness_model \
    frozen/robustness_kernels.json \
    frozen/generation_audit_robustness.json \
    configs/llm_study_paper_robustness.json \
    results/llm_study_paper_robustness
  echo "== Prespecified robustness checkpoint 2: ibm-granite/granite-4.2-8b =="
  run_model_chain robustness_model_2 \
    frozen/robustness2_kernels.json \
    frozen/generation_audit_robustness2.json \
    configs/llm_study_paper_robustness2.json \
    results/llm_study_paper_robustness2
fi

if [[ "${TIS_RUN_SCALE:-0}" == "1" ]]; then
  echo "== Post-hoc scale replication 1: mistralai/Mistral-Small-24B-Instruct-2501 =="
  run_model_chain scale_model_1 \
    frozen/scale1_kernels.json \
    frozen/generation_audit_scale1.json \
    configs/llm_study_paper_scale1.json \
    results/llm_study_paper_scale1 \
    32
  echo "== Post-hoc scale replication 2: Qwen/Qwen3-32B =="
  run_model_chain scale_model_2 \
    frozen/scale2_kernels.json \
    frozen/generation_audit_scale2.json \
    configs/llm_study_paper_scale2.json \
    results/llm_study_paper_scale2 \
    16
  echo "== Post-hoc scale replication 3: zai-org/GLM-4-32B-0414 =="
  run_model_chain scale_model_3 \
    frozen/scale3_kernels.json \
    frozen/generation_audit_scale3.json \
    configs/llm_study_paper_scale3.json \
    results/llm_study_paper_scale3 \
    16
fi

echo "== COMPLETE =="
echo "Primary kernels:      frozen/qwen_kernels.json -> results/llm_study_paper"
if [[ "${TIS_SKIP_ROBUSTNESS:-0}" != "1" ]]; then
  echo "Robustness kernels 1: frozen/robustness_kernels.json -> results/llm_study_paper_robustness"
  echo "Robustness kernels 2: frozen/robustness2_kernels.json -> results/llm_study_paper_robustness2"
fi
echo "Run log: $RUN_LOG"
