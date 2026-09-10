#!/usr/bin/env bash
# Reproduces the headline results end-to-end. See README.md for details.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. Prep brand data (real if data/raw/twcs.csv exists, else synthetic) =="
python3 -m src.data_prep

echo "== 2. (Re)build golden-set template if it doesn't exist =="
if [ ! -f eval/golden_eval.csv ]; then
  echo "NOTE: eval/golden_eval.csv not found."
  echo "If this is your first run with real data, generate + hand-label it:"
  echo "    python3 scripts/make_golden_set.py"
  echo "    <label eval/golden_eval_TEMPLATE.csv, save as eval/golden_eval.csv>"
  echo "Falling back to whatever golden_eval.csv is currently checked in (demo labels)."
fi

echo "== 3. Run evaluation harness (trivial baseline / simple baseline / agent) =="
python3 -m src.evaluate

echo "== Done. See eval/results.csv =="
