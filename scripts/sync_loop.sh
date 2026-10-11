#!/bin/bash
# Periodically commit+push logs/ and results/ to GitHub. Runs until stopped.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${1:-120}"
cd "$REPO"
while true; do
  sleep "$INTERVAL"
  git add -A logs results src scripts README.md requirements.txt .gitignore 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "auto-sync logs+results $(date '+%F %T')" || true
  fi
  git push -q origin main 2>>logs/sync_err.log || true
done
