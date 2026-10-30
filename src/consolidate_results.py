"""Builds the canonical pooled nested-CV results (first pass and final run) for all 280 configurations.

The original runs are in results/ (first pass) and results_uncapped/ (final run of SVM-RBF, RF, XGBoost
and LightGBM). Kettle and microwave were re-run after the channel-map correction: their corrected rows are in
results_fixed_first_pass/ and results_fixed_final/ and replace the original rows. LR, DT and k-NN were not
re-run in the final run, so their final values are the (corrected) first-pass values.
Outputs: results_final/pooled_first_pass_280.csv, results_final/pooled_final_280.csv
"""
import os, pandas as pd, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P = lambda *a: os.path.join(ROOT, *a)
K = ["appliance", "arm", "model", "imbalance"]
COLS = K + ["status", "n_rows", "row_capped", "f1_mean", "f1_std", "precision_mean", "recall_mean", "accuracy_mean", "accuracy_std", "elapsed_seconds"]
FIXED = ["kettle", "microwave"]

def with_accuracy(summary, folds):
    f = pd.read_csv(folds); f["acc"] = (f.tn + f.tp) / (f.tn + f.fp + f.fn + f.tp)
    a = f.groupby(K).acc.agg(accuracy_mean="mean", accuracy_std="std").reset_index()
    s = pd.read_csv(summary).drop(columns=[c for c in ["accuracy_mean", "accuracy_std"] if c in pd.read_csv(summary).columns])
    return s.merge(a, on=K, how="left")

def merge(base, fixed):
    base = base[~base.appliance.isin(FIXED)]
    return pd.concat([base, fixed[fixed.appliance.isin(FIXED)]], ignore_index=True)

first = merge(pd.read_csv(P("results", "summary_results_with_accuracy.csv")),
              with_accuracy(P("results_fixed_first_pass", "summary_results.csv"), P("results_fixed_first_pass", "fold_results.csv")))
rerun = merge(pd.read_csv(P("results_uncapped", "summary_results_with_accuracy.csv")),
              with_accuracy(P("results_fixed_final", "summary_results.csv"), P("results_fixed_final", "fold_results.csv")))
final = pd.concat([rerun, first[first.model.isin(["LogisticRegression", "DecisionTree", "kNN"])]], ignore_index=True)
os.makedirs(P("results_final"), exist_ok=True)
for name, d in [("pooled_first_pass_280.csv", first), ("pooled_final_280.csv", final)]:
    d = d[[c for c in COLS if c in d.columns]].sort_values(K).reset_index(drop=True)
    assert len(d) == 280 and not d.duplicated(K).any(), (name, len(d))
    d.to_csv(P("results_final", name), index=False); print(name, len(d), "fitted:", int((d.status == "ok").sum()))
