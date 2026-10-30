"""
Direct capped-vs-uncapped comparison for the 4 models that were re-run without
(or with a revised) row cap: SVM_RBF, RandomForest, XGBoost, LightGBM.

Joins results/summary_results_with_accuracy.csv (the original capped sweep) against
results_uncapped/summary_results_with_accuracy.csv (the uncapped/re-capped re-run)
on (appliance, arm, model, imbalance), and reports, per combo:
  - capped vs uncapped F1 (mean), accuracy (mean), elapsed_seconds
  - the delta (uncapped - capped) for F1 and accuracy
  - what each condition's row cap actually was (for context: SVM_RBF/RandomForest/
    XGBoost/LightGBM all had a historical-first-pass cap; in the "uncapped" condition,
    RandomForest/XGBoost/LightGBM trained on the full pooled data, while SVM_RBF was
    re-capped at 100,000 rows with stratified sampling -- see model_pipeline.py and
    the project notes for why)

Also produces two rollups:
  - per-model and per-appliance mean delta F1 / delta accuracy across all their
    combos (the "typical" effect of uncapping)
  - a "best config per appliance" comparison: for each appliance, the best F1 under
    the ORIGINAL fully-capped condition vs the best F1 once the 4 re-run models use
    their uncapped/re-capped numbers and the other 3 models (LogisticRegression,
    DecisionTree, kNN -- never capped in either condition) keep their original
    numbers unchanged. Reports whether the winning (model, arm, imbalance) changed.

Writes results_uncapped/capped_vs_uncapped_comparison.csv (per-combo) and prints the
rollups to stdout. Pure Python stdlib -- no pandas dependency, so it runs anywhere
(including the device_bash bridge, which has previously lacked the ML packages the
real Mac has). Never touches results/ or results_uncapped/'s own summary/fold files.
"""
import csv
import statistics
from collections import defaultdict

CAPPED_PATH = "results/summary_results_with_accuracy.csv"
UNCAPPED_PATH = "results_uncapped/summary_results_with_accuracy.csv"
OUT_PATH = "results_uncapped/capped_vs_uncapped_comparison.csv"

RECAPPED_MODELS = {"SVM_RBF", "RandomForest", "XGBoost", "LightGBM"}
HISTORICAL_CAP = {"SVM_RBF": 30_000, "RandomForest": 60_000, "XGBoost": 100_000, "LightGBM": 100_000}
UNCAPPED_CAP_NOTE = {"SVM_RBF": "100,000 (stratified)", "RandomForest": "none (full pooled)",
                     "XGBoost": "none (full pooled)", "LightGBM": "none (full pooled)"}


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            key = (r["appliance"], r["arm"], r["model"], r["imbalance"])
            rows[key] = r
    return rows


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    capped = load(CAPPED_PATH)
    uncapped = load(UNCAPPED_PATH)

    per_combo = []
    for key, u in uncapped.items():
        c = capped.get(key)
        if c is None:
            continue  # shouldn't happen -- every re-run combo existed in the original sweep
        appliance, arm, model, imbalance = key
        cf1, uf1 = f(c["f1_mean"]), f(u["f1_mean"])
        cacc, uacc = f(c.get("accuracy_mean")), f(u.get("accuracy_mean"))
        cel, uel = f(c["elapsed_seconds"]), f(u["elapsed_seconds"])
        per_combo.append({
            "appliance": appliance, "arm": arm, "model": model, "imbalance": imbalance,
            "capped_row_cap": HISTORICAL_CAP.get(model), "uncapped_row_cap": UNCAPPED_CAP_NOTE.get(model),
            "capped_f1_mean": cf1, "uncapped_f1_mean": uf1,
            "delta_f1": (uf1 - cf1) if (cf1 is not None and uf1 is not None) else None,
            "capped_accuracy_mean": cacc, "uncapped_accuracy_mean": uacc,
            "delta_accuracy": (uacc - cacc) if (cacc is not None and uacc is not None) else None,
            "capped_elapsed_seconds": cel, "uncapped_elapsed_seconds": uel,
        })

    per_combo.sort(key=lambda r: (r["appliance"], r["arm"], r["model"], r["imbalance"]))
    with open(OUT_PATH, "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=list(per_combo[0].keys()))
        w.writeheader()
        w.writerows(per_combo)
    print(f"Wrote {OUT_PATH} -- {len(per_combo)} matched combos\n")

    # Rollup 1: mean delta F1 / accuracy per model
    print("=== Mean delta F1 / accuracy per model (uncapped - capped) ===")
    by_model = defaultdict(list)
    for r in per_combo:
        by_model[r["model"]].append(r)
    for model in sorted(by_model):
        rs = by_model[model]
        d_f1 = [r["delta_f1"] for r in rs if r["delta_f1"] is not None]
        d_acc = [r["delta_accuracy"] for r in rs if r["delta_accuracy"] is not None]
        print(f"  {model:14s} n={len(rs):3d}  mean_delta_f1={statistics.mean(d_f1):+.4f}"
              f"  mean_delta_accuracy={statistics.mean(d_acc):+.4f}" if d_acc else
              f"  {model:14s} n={len(rs):3d}  mean_delta_f1={statistics.mean(d_f1):+.4f}")

    # Rollup 2: mean delta F1 per appliance (across the 4 re-run models only)
    print("\n=== Mean delta F1 per appliance (4 re-run models only) ===")
    by_appliance = defaultdict(list)
    for r in per_combo:
        by_appliance[r["appliance"]].append(r["delta_f1"])
    for appliance in sorted(by_appliance):
        vals = [v for v in by_appliance[appliance] if v is not None]
        print(f"  {appliance:16s} n={len(vals):3d}  mean_delta_f1={statistics.mean(vals):+.4f}"
              f"  min={min(vals):+.4f}  max={max(vals):+.4f}")

    # Rollup 3: best config per appliance, capped-sweep vs "final" (recapped models'
    # new numbers + the 3 never-capped models' original numbers, unchanged)
    print("\n=== Best config per appliance: original capped sweep vs final (uncapped/re-capped) ===")
    # capped: best across ALL 7 models' original numbers
    capped_best = defaultdict(lambda: (-1, None))
    for key, c in capped.items():
        appliance = key[0]
        v = f(c["f1_mean"])
        if v is not None and v > capped_best[appliance][0]:
            capped_best[appliance] = (v, key)

    # final: for re-run models use the new uncapped numbers; for the other 3, keep original
    final_pool = dict(capped)  # start from capped, then overlay the 4 re-run models
    final_pool.update(uncapped)
    final_best = defaultdict(lambda: (-1, None))
    for key, r in final_pool.items():
        appliance = key[0]
        v = f(r["f1_mean"])
        if v is not None and v > final_best[appliance][0]:
            final_best[appliance] = (v, key)

    for appliance in sorted(capped_best):
        cv, ck = capped_best[appliance]
        fv, fk = final_best[appliance]
        changed = "CHANGED" if ck != fk else "same"
        print(f"  {appliance:16s} capped_best={cv:.4f} {ck[1:]}  ->  final_best={fv:.4f} {fk[1:]}  [{changed}]")


if __name__ == "__main__":
    main()
