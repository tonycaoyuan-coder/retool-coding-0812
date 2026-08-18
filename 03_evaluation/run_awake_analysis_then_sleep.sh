#!/bin/zsh
set -eu

ROOT="${0:A:h:h}"
PYTHON="$ROOT/.venv/bin/python"
GUARDIAN="$ROOT/03_evaluation/wait_analyze_then_sleep.py"

/usr/bin/caffeinate -dimsu "$PYTHON" "$GUARDIAN"

if ! /usr/bin/pmset sleepnow; then
  /usr/bin/osascript -e 'tell application "System Events" to sleep'
fi
