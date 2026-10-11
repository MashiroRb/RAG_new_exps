#!/bin/bash
# Offline experiment pipeline runner. All phases write logs under RAG_new_exps/logs.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="$ROOT/RAG_new_exps"
PY=python3
TS="$(date +%Y%m%d_%H%M%S)"
STATUS="$REPO/results/status.txt"
echo "run_all started $TS" > "$STATUS"

run_phase() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] phase $name START" >> "$STATUS"
  "$PY" "$@" > "$REPO/logs/${name}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] phase $name EXIT=$rc" >> "$STATUS"
  return 0
}

run_phase parse_logs        "$REPO/src/parse_logs.py"
run_phase infer_injection   "$REPO/src/infer_injection.py"
run_phase replay_alignment  "$REPO/src/replay_alignment.py"
run_phase r1_metrics        "$REPO/src/r1_metrics.py"
run_phase r2_kappa          "$REPO/src/r2_kappa.py"
run_phase r3_grid           "$REPO/src/r3_grid.py"
run_phase r4_anchor         "$REPO/src/r4_anchor.py"
echo "[$(date '+%F %T')] run_all DONE" >> "$STATUS"
