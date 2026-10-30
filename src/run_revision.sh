#!/bin/bash
# Revision runs: calibrated CNN and data-quality sensitivity in parallel, then the revision analysis.
# Keeps the Mac awake, resumes after any stop (at most 5 restarts per phase). Run from anywhere:
#   bash "<path to nilm>/src/run_revision.sh"
cd "$(dirname "$0")" || exit 1
OUT="../results_all_experiments"; mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 PYTORCH_ENABLE_MPS_FALLBACK=1
run_phase () {  # $1 = CNN or SENS
  n=0
  until grep -q "PHASE R_$1 DONE" "$OUT/logs/phase_R_$1.log" 2>/dev/null || [ $n -ge 5 ]; do
    python3 run_revision_experiments.py --phase "$1" >> "$OUT/logs/phase_R_$1.log" 2>&1
    n=$((n+1)); sleep 10
  done
}
caffeinate -dimsu -w $$ &
( run_phase CNN ) &
( run_phase SENS ) &
wait
python3 analyse_revision.py > "$OUT/logs/revision_analysis.txt" 2>&1
python3 -m pip freeze > ../requirements-lock.txt 2>/dev/null
python3 --version >> ../requirements-lock.txt 2>&1
echo "ALL REVISION RUNS DONE $(date)" | tee -a "$OUT/logs/revision_done.log"
