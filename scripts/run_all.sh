#!/bin/bash
# Offline experiment pipeline runner (new controlled experiments, no LLM API).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO="$ROOT/RAG_new_exps"
PY=python3
STATUS="$REPO/results/status.txt"
echo "run_all started $(date '+%F %T')" > "$STATUS"

run_phase() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] phase $name START" >> "$STATUS"
  "$PY" "$@" > "$REPO/logs/${name}.log" 2>&1
  echo "[$(date '+%F %T')] phase $name EXIT=$?" >> "$STATUS"
  return 0
}

run_phase r1_metrics      "$REPO/src/r1_metrics.py"
run_phase r4_anchor       "$REPO/src/r4_anchor.py"
run_phase r2_kappa        "$REPO/src/r2_kappa.py"
run_phase e0_calibration  "$REPO/src/e0_calibration.py" camera
run_phase e0_calibration_ev "$REPO/src/e0_calibration.py" ev
run_phase e1_filtering    "$REPO/src/e1_filtering.py"
run_phase e2_rho          "$REPO/src/e2_rho.py"
run_phase e3_nstatic      "$REPO/src/e3_nstatic.py"
run_phase r1p_metric_stability "$REPO/src/r1p_metric_stability.py"
echo "[$(date '+%F %T')] run_all DONE" >> "$STATUS"
