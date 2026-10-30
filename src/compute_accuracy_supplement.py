"""
Supplementary, read-only pass: derives an accuracy_mean / accuracy_std column
for every completed combo directly from the tn/fp/fn/tp confusion-matrix
counts that nested_cv_evaluate() already writes to fold_results.csv for every
outer fold. No rerun of any model is needed -- accuracy = (tp+tn)/(tp+tn+fp+fn)
is fully computable from data already on disk.

This deliberately does NOT touch fold_results.csv or summary_results.csv in
place. It writes a new file, summary_results_with_accuracy.csv, alongside
them in the same --dir. Rationale: results_uncapped/ is being appended to
live by an unattended multi-day run on this machine (RandomForest is still
mid-sweep as of 2026-09-18) -- editing the schema of a file a running process
is actively appending to would corrupt it (later rows would gain columns the
already-written header doesn't have). Re-run this script any time (safe,
idempotent, cheap) -- once after RandomForest finishes and once after the
full XGBoost/LightGBM/SVM_RBF chain completes, to get the final version.

USAGE
    python3 compute_accuracy_supplement.py --dir ../results
    python3 compute_accuracy_supplement.py --dir ../results_uncapped
"""
import argparse
import csv
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    fold_path = os.path.join(args.dir, "fold_results.csv")
    summary_path = os.path.join(args.dir, "summary_results.csv")
    out_path = os.path.join(args.dir, "summary_results_with_accuracy.csv")

    # key -> list of per-fold accuracy values
    acc_by_combo = {}
    with open(fold_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["appliance"], row["arm"], row["model"], row["imbalance"])
            tn, fp, fn, tp = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])
            denom = tn + fp + fn + tp
            if denom == 0:
                continue
            acc_by_combo.setdefault(key, []).append((tp + tn) / denom)

    combo_stats = {}
    for key, accs in acc_by_combo.items():
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else ""  # pandas .std() on n=1 is NaN too
        combo_stats[key] = (mean_acc, std_acc, len(accs))

    n_matched, n_rows = 0, 0
    with open(summary_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames) + ["accuracy_mean", "accuracy_std", "accuracy_n_folds"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            n_rows += 1
            key = (row["appliance"], row["arm"], row["model"], row["imbalance"])
            if key in combo_stats:
                mean_acc, std_acc, n_folds = combo_stats[key]
                row["accuracy_mean"] = mean_acc
                row["accuracy_std"] = std_acc
                row["accuracy_n_folds"] = n_folds
                n_matched += 1
            else:
                row["accuracy_mean"] = ""
                row["accuracy_std"] = ""
                row["accuracy_n_folds"] = ""
            writer.writerow(row)

    print(f"{args.dir}: wrote {out_path} -- {n_matched}/{n_rows} summary rows matched to fold-level "
          f"accuracy (unmatched rows are status != ok, e.g. not_applicable, which have no fold data).")


if __name__ == "__main__":
    main()
