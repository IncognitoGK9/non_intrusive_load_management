#!/bin/bash
# Keeps the Mac awake (caffeinate) and resumes until all 600 folds are done.
cd "$(dirname "$0")" || exit 1
OUT="../results_lho_factorial"
until grep -q "ALL DONE" "$OUT/run.log" 2>/dev/null; do
  caffeinate -dimsu python3 run_lho_factorial.py --plan "$OUT/plan_factorial.json" --partition "$OUT/house_partition.csv" --features-dir ../data/features --out "$OUT" >> "$OUT/run.log" 2>&1
  sleep 5
done
