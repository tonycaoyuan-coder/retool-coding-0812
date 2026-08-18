#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h:h}"
cd "$ROOT_DIR"

export PYTHONPATH="src"
mkdir -p artifacts/evaluation-posthoc/supervisor

.venv/bin/python -u 03_evaluation/posthoc.py run --resume
.venv/bin/python -u 03_evaluation/posthoc.py finalize
