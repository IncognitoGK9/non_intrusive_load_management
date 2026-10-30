#!/bin/bash
# Runs all remaining experiments in three parallel chains, keeps the Mac awake, and resumes after any stop.
cd "$(dirname "$0")" || exit 1
OUT="../results_all_experiments"; mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
run_phase () {  # $1 = phase letter
  until grep -q "PHASE $1 DONE" "$OUT/logs/phase_$1.log" 2>/dev/null; do
    python3 run_all_experiments.py --phase "$1" >> "$OUT/logs/phase_$1.log" 2>&1
    sleep 10
  done
}
caffeinate -dimsu -w $$ &
( run_phase A; run_phase E ) &
( run_phase B ) &
( run_phase C; run_phase D ) &
wait
echo "ALL PHASES DONE" >> "$OUT/logs/all_done.log"
