#!/bin/bash
# Status of the revision runs.  Usage:  bash revision_status.sh          (one report)
#                                       bash revision_status.sh --watch  (refreshes every 60 s; Ctrl+C to stop)
cd "$(dirname "$0")" || exit 1
report () {
python3 - <<'EOF'
import os, csv, time, subprocess
R = "../results_all_experiments"; L = R + "/logs"
def rows(p):
    try:
        with open(p) as f: return list(csv.DictReader(f))
    except FileNotFoundError: return []
def bar(n, tot, w=30):
    k = int(w * n / tot) if tot else 0; return "[" + "#" * k + "-" * (w - k) + f"] {n}/{tot} ({100*n/tot:.0f}%)"
def eta(rs, tot):
    if not rs: return "not started"
    s = sum(float(r.get("elapsed_seconds") or 0) for r in rs); left = (tot - len(rs)) * s / len(rs)
    return "finished" if len(rs) >= tot else f"about {int(left//3600)} h {int(left%3600//60)} min left"
def last(log):
    try:
        lines = [l.strip() for l in open(log, errors="ignore") if l.startswith("20")]
        return lines[-1] if lines else "(no progress line yet)"
    except FileNotFoundError: return "(log not found)"
def errors(log):
    try: return sum(1 for l in open(log, errors="ignore") if "Traceback" in l or "Error" in l and "Warning" not in l)
    except FileNotFoundError: return 0
running = subprocess.run(["pgrep", "-f", "run_revision_experiments.py"], capture_output=True, text=True).stdout.split()
print("=" * 72); print("REVISION RUNS  " + time.strftime("%Y-%m-%d %H:%M:%S")); print("=" * 72)
for name, ph, tot in [("Calibrated CNN", "R_CNN", 50), ("Data-quality sensitivity", "R_SENS", 805)]:
    rs = rows(f"{R}/phase_{ph}/fold_results.csv"); log = f"{L}/phase_{ph}.log"
    done = os.path.exists(log) and f"PHASE {ph} DONE" in open(log, errors="ignore").read()
    print(f"\n{name}\n  {bar(len(rs), tot)}   {'DONE' if done else eta(rs, tot)}")
    print(f"  last: {last(log)[:110]}")
    e = errors(log)
    if e: print(f"  WARNING: {e} error line(s) in {log}; the script restarts itself up to 5 times")
    if ph == "R_CNN" and rs:
        for v in ["centred", "causal"]:
            x = [float(r["f1"]) for r in rs if r["variant"] == v]
            if x: print(f"  {v:8s}: {len(x):2d}/25 folds, mean F1 so far {sum(x)/len(x):.3f}")
print("\nPython workers running:", len(running) if running else "none")
if os.path.exists(f"{L}/revision_done.log"):
    print("\n*** ALL DONE. Tell Claude: done ***")
elif not running:
    print("\nNothing is running and the runs are not finished. Restart with:\n  nohup bash run_revision.sh > ../results_all_experiments/logs/revision_master.log 2>&1 &")
print("=" * 72)
EOF
}
if [ "$1" == "--watch" ]; then
  while true; do clear; report; sleep 60; done
else
  report
fi
