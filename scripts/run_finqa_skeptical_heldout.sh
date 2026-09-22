#!/usr/bin/env bash
# FinQA skeptical-workflow HELD-OUT calibrations and audits.
# Runs ONLY after the development gate in configs/finqa_skeptical_gate.json
# passes (the caller checks; this script also refuses without the marker).
# Reuses the frozen held-out panel and banks; ~35 minutes on one H100.
set -eo pipefail
cd "$HOME/tis"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/tis/.hf"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
PY="$HOME/tis/.llm-venv/bin/python"
if [ ! -f frozen/finqa_skeptical_gate_pass.json ]; then
  echo "[abort] gate-pass marker frozen/finqa_skeptical_gate_pass.json absent"
  exit 1
fi
echo "[start $(date -Iseconds)] host=$(hostname)"
for proto in "configs/finqa_protocol.json:" "configs/finqa_protocol_phi.json:_phi"; do
  cfg="${proto%%:*}"; suffix="${proto##*:}"
  if [ ! -f "frozen/finqa_kernels_heldout2${suffix}_skeptical.json" ]; then
    "$PY" -m tis.finqa.cli calibrate \
      --protocol "$cfg" \
      --panel-jsonl frozen/finqa_panel_heldout2.jsonl \
      --banks frozen/finqa_banks_heldout2.json \
      --output "frozen/finqa_kernels_heldout2${suffix}_skeptical.json" \
      --workflow skeptical
  fi
  if [ ! -f "frozen/finqa_audit_heldout2${suffix}_skeptical.json" ]; then
    "$PY" -m tis.finqa.cli audit-generations \
      --artifact "frozen/finqa_kernels_heldout2${suffix}_skeptical.json" \
      --protocol "$cfg" \
      --panel-jsonl frozen/finqa_panel_heldout2.jsonl \
      --banks frozen/finqa_banks_heldout2.json \
      --output "frozen/finqa_audit_heldout2${suffix}_skeptical.json"
  fi
done
echo "[all_done $(date -Iseconds)]"
