"""Regenerates the leave-house-out plans: for every configuration, the hyperparameters most often selected
across the five outer folds of its pooled nested CV (final run for SVM-RBF, RF, XGBoost, LightGBM; first pass
for LR, DT, k-NN; corrected kettle and microwave runs replace the original rows).
Writes results_lho_factorial/plan_factorial.json (LR, DT, XGBoost) and results_all_experiments/config/plan_{A,B,E}.json."""
import os, ast, json, collections, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); P = lambda *a: os.path.join(ROOT, *a)
FIX = ["kettle", "microwave"]
def folds(orig, fixed):
    o = pd.read_csv(P(orig, "fold_results.csv")); f = pd.read_csv(P(fixed, "fold_results.csv"))
    return pd.concat([o[~o.appliance.isin(FIX)], f[f.appliance.isin(FIX)]], ignore_index=True)
first = folds("results", "results_fixed_first_pass"); final = folds("results_uncapped", "results_fixed_final")
SRC = {m: final for m in ["SVM_RBF", "RandomForest", "XGBoost", "LightGBM"]}; SRC.update({m: first for m in ["LogisticRegression", "DecisionTree", "kNN"]})
def plan(models):
    out = []
    for m in models:
        f = SRC[m]; f = f[f.model == m]
        for (a, arm, imb), s in f.groupby(["appliance", "arm", "imbalance"]):
            ps = [json.dumps({k.replace("model__", ""): v for k, v in ast.literal_eval(x).items()}, sort_keys=True) for x in s.best_params]
            out.append(dict(appliance=a, arm=arm, model=m, imbalance=imb, params=json.loads(collections.Counter(ps).most_common(1)[0][0])))
    return out
os.makedirs(P("results_all_experiments", "config"), exist_ok=True)
for path, models in [(P("results_lho_factorial", "plan_factorial.json"), ["LogisticRegression", "DecisionTree", "XGBoost"]),
                     (P("results_all_experiments", "config", "plan_A.json"), ["LightGBM", "RandomForest", "kNN"]),
                     (P("results_all_experiments", "config", "plan_B.json"), ["LogisticRegression", "DecisionTree", "XGBoost", "LightGBM"]),
                     (P("results_all_experiments", "config", "plan_E.json"), ["SVM_RBF"])]:
    p = plan(models); json.dump(p, open(path, "w")); print(os.path.relpath(path, ROOT), len(p))
