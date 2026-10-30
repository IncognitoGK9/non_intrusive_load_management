"""
Classical-ML model sweep + imbalance-handling ablation + literature comparison,
for the classical-ML NILM ON/OFF classification study (REFIT, 5 canonical
appliances, weighted-F1 evaluation matching Tanoni et al. 2022 / Pirnat et al.
2024 so results are directly comparable).

Model set (7, per user request to include simple baselines -- 2026-09-15):
spans simple linear (Logistic Regression) and single-tree (Decision Tree)
baselines, an instance-based method (kNN), a kernel method (SVM), and three
tree-ensemble methods (Random Forest, XGBoost, LightGBM). An MLP (small
neural network) is deliberately NOT included here -- it isn't a classical
model, and lumping it in would blur the classical-vs-deep-learning line the
whole study is built on. It's reserved as the LIGHTEST_DL_CANDIDATE for the
Phase 4 exploratory hybrid step instead (see paper-draft-notes.md).

This module is imported by run_experiment.py, which does the actual sweep
against real REFIT feature tables (built by build_features.py). Running this
file directly just prints the literature table and a tiny self-test on
synthetic data -- it does not need real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

MODELS = {
    "LogisticRegression": (Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(
                                max_iter=1000, random_state=RANDOM_STATE))]),
                            {"lr__C": [0.1, 1, 10]}),
    "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                      {"max_depth": [6, 12, None], "min_samples_leaf": [1, 5]}),
    "kNN": (Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())]),
            {"knn__n_neighbors": [3, 5, 11]}),
    "SVM_RBF": (Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel="rbf"))]),
                {"svc__C": [1, 10], "svc__gamma": ["scale", "auto"]}),
    # n_jobs=1 here deliberately: GridSearchCV below already parallelises across
    # the grid (n_jobs capped from -1, currently 3 on this machine -- see the
    # comment in nested_cv_evaluate) and outer folds -- also letting the forest
    # itself spawn threads oversubscribes the CPU and was measured to slow
    # things down a lot (self-test on 40k rows: LogisticRegression finished
    # all 4 imbalance strategies before RandomForest+class_weight finished
    # even one).
    #
    # max_depth=None (unbounded) REMOVED from this grid on 2026-09-17, during
    # the uncapped re-run, on an 8GB machine: an unbounded RandomForest is 300
    # trees, each grown to full depth on a pooled fold of over a million rows
    # (fridge arm A: n_rows=1,080,000) -- unlike a single unbounded
    # DecisionTree (still fine, see above), the cumulative memory of 300 such
    # trees repeatedly triggered numpy/sklearn ArrayMemoryError /
    # safe_realloc MemoryError mid-fit even after n_jobs was capped at 2.
    # GridSearchCV silently NaN's a failed fit and still returns a result from
    # the survivors, so this wasn't corrupting completed combos, but it was
    # both slow (55-125 min/combo) and a live crash risk. This is a disclosed
    # change to the tuning search space itself (not just an engineering
    # trade-off like n_jobs), so it must be reported in the manuscript's
    # methodology alongside the row-cap history: RandomForest's uncapped
    # results were tuned over max_depth=12 only, not {None, 12} like the
    # original capped sweep. The two fridge/arm-A combos already run under the
    # old {None, 12} grid (none, class_weight) were deleted from
    # results_uncapped/ and re-run under this constrained grid for internal
    # consistency across the sweep.
    "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                      {"n_estimators": [100, 300], "max_depth": [12]}),
    # n_jobs=1 for the same oversubscription reason as RandomForest above --
    # GridSearchCV's own n_jobs (see nested_cv_evaluate) is where the
    # parallelism should live.
    "XGBoost": (xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
                {"n_estimators": [100, 300], "max_depth": [4, 8]}),
    "LightGBM": (lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
                 {"n_estimators": [100, 300], "max_depth": [-1, 8]}),
}

# Row caps for models whose training cost scales badly with row count --
# enforced in run_experiment.py, not here (this module only defines the
# policy; run_experiment.py applies it per combo and logs when it kicks in).
# A documented, stated choice, not a silent shortcut.
#
# Added/expanded 2026-09-16 after real REFIT runs on the user's machine: with
# only SVM_RBF capped (at 30,000), a single kettle appliance (arm A, the
# smaller of the two arms) took ~51 minutes for just 16 of 56 combos, with
# the still-uncapped RandomForest/XGBoost/LightGBM combos on the full
# ~840,000-row pooled kettle dataset still ahead -- projected to add hours
# more for kettle alone, and days across all 5 appliances x 2 arms (fridge,
# the appliance with the most monitoring houses, would be the worst case).
# The user asked for the tree ensembles capped too, to make the full sweep
# finish in a tractable time on a single machine.
#
# Cap sizes differ per model because their training cost scales differently
# with row count, not out of an arbitrary preference for round numbers:
# SVM_RBF's cost grows roughly quadratically with rows (hence the smallest
# cap, unchanged from before); scikit-learn's RandomForest is markedly
# slower per row than the other two ensembles at this scale (observed
# directly: ~61s for one combo on 40k synthetic rows even after fixing the
# n_jobs oversubscription bug above, versus XGBoost/LightGBM's histogram-
# based implementations, which are typically much faster per row); XGBoost
# and LightGBM get the largest cap for that reason. If a capped result ever
# looks borderline important to the paper's conclusion, it can be re-run
# uncapped for a final check -- this cap is about making the FIRST full
# pass across all appliances tractable, not a permanent ceiling.
# HISTORICAL: these were the row caps used for the first full pass across all
# appliances (2026-09-16/17) -- see results/summary_results.csv (row_capped=True
# rows) and run_log.txt for that run's real numbers, and the manuscript's
# Section 5.7 / Section 8 for the disclosed rationale. Kept here for the record
# and for the capped-vs-uncapped comparison in the manuscript; NOT applied below.
ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS = {
    "SVM_RBF": 30_000,
    "RandomForest": 60_000,
    "XGBoost": 100_000,
    "LightGBM": 100_000,
}

# UNCAPPED RE-RUN (decided 2026-09-17, after reviewing the first pass's capped
# numbers): every model now trains on the FULL pooled dataset for its
# appliance/arm -- no subsampling. Emptying this dict (rather than deleting it)
# keeps run_experiment.py's existing "ROW_CAP_BY_MODEL.get(model_name)" lookup
# working unchanged: it now returns None for every model, i.e. no cap. Point
# run_experiment.py at a fresh --out-dir (e.g. results_uncapped/) for this run
# so the original capped results in results/ are never overwritten.
# SESSION 7/8 (2026-09-19): SVM_RBF-only cap, reintroduced after the full uncapped
# attempt was tried and abandoned. Real evidence: SVM_RBF was launched fully uncapped
# (full pooled rows, 840k-1,140,000 depending on appliance) on the Mac (18GB RAM,
# n_jobs=3) and left running for ~9 hours -- ZERO of the 40 SVM_RBF combos completed
# in that time (not even the first, fridge/armA/none). RandomForest/XGBoost/LightGBM
# all completed uncapped without incident, so this is specific to SVM_RBF's kernel
# method: sklearn's SVC(kernel="rbf") scales roughly quadratically-to-cubically in
# the number of training rows (libsvm), unlike the near-linear tree ensembles/boosted
# methods. Going from the original 30,000-row cap to the full ~1M rows is a ~30-38x
# increase in n -- at quadratic scaling alone that's roughly 900-1450x slower, which
# is consistent with 9 hours not being enough for even one combo (the equivalent
# 30k-row combo took as little as 6-300s). This is a disclosed, real finding, not an
# implementation bug: SVM_RBF does not scale to REFIT's real per-appliance row counts
# within a practical compute budget, full stop.
#
# Rather than re-run at the ORIGINAL 30,000-row cap (which would just reproduce the
# historical-first-pass numbers with nothing new to report), we chose 100,000 rows --
# ~3.3x the original cap, giving SVM_RBF meaningfully more data than before while
# staying tractable: at quadratic-scaling extrapolation from the real 30k-cap timings,
# the worst historical combo (kettle/armA/ADASYN, 812.2s) projects to roughly
# 812.2 * (100000/30000)**2 =~ 9,000s (2.5 hours) for the single worst combo, with
# most combos far faster -- a full 40-combo sweep should complete in well under a day,
# not another open-ended multi-day gamble. This is a genuinely different, disclosed
# methodological choice from the original 30k cap and must be reported as such (see
# ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS below for the original value).
#
# Also switched (see run_experiment.py's row-cap sampling code) from pure random
# subsampling to STRATIFIED subsampling for whichever model(s) this dict caps, so the
# capped subsample preserves the real class balance rather than risking a random draw
# that under/over-represents the already-rare positive class (microwave 0.27% ON,
# kettle 0.71% ON) -- the original historical-first-pass cap used plain random
# sampling; this is a deliberate improvement on that, and both are disclosed.
ROW_CAP_BY_MODEL = {"SVM_RBF": 100_000}
SLOW_MODELS = set(ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS.keys())

# ---------------------------------------------------------------------------
# PROTOCOL SELECTION (added 2026-09-23)
#
# This study reports two experimental protocols, and previously each one was
# obtained by hand-editing ROW_CAP_BY_MODEL and RandomForest's max_depth grid
# in this file between runs. That made the two runs impossible to reproduce
# from the code alone -- the file only ever holds the settings of whichever
# run happened last -- and it is how RandomForest's search space came to
# differ between the two published runs without that difference being
# controlled for. Both protocols are now declared here and selected at run
# time with run_experiment.py --protocol, so a reader can reproduce either.
#
#   first_pass    the capped sweep in results/. Per-model row caps as in
#                 ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS, PURE RANDOM
#                 subsampling, and RandomForest tuned over max_depth in
#                 {None, 12}.
#
#   final         the run in results_uncapped/. Tree ensembles uncapped,
#                 SVM_RBF capped at 100,000 rows with STRATIFIED subsampling,
#                 and RandomForest tuned over max_depth = 12 only (unbounded
#                 removed because 300 fully grown trees on >1M rows exhausted
#                 memory; see the note above).
#
#   depth_control the control run that separates those two changes. Identical
#                 to first_pass in every respect -- same 60,000-row caps, same
#                 pure random subsampling -- EXCEPT that RandomForest is tuned
#                 over max_depth = 12 only, as in final. Comparing
#                 depth_control against first_pass therefore isolates the
#                 effect of restricting tree depth at a fixed data size, and
#                 comparing depth_control against final isolates the effect of
#                 the data increase at a fixed search space. Neither
#                 comparison is available from the two published runs alone,
#                 because they differ in both respects at once.
# ---------------------------------------------------------------------------
PROTOCOLS: dict[str, dict] = {
    "first_pass": {
        "row_caps": dict(ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS),
        "cap_sampling": "random",
        "rf_max_depth": [None, 12],
    },
    "final": {
        "row_caps": {"SVM_RBF": 100_000},
        "cap_sampling": "stratified",
        "rf_max_depth": [12],
    },
    "depth_control": {
        "row_caps": dict(ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS),
        "cap_sampling": "random",
        "rf_max_depth": [12],
    },
}

# Mutated in place by apply_protocol() so that modules which did
# "from model_pipeline import ROW_CAP_BY_MODEL" still see the active settings.
ACTIVE_PROTOCOL: dict = {"name": "final", **{k: v for k, v in PROTOCOLS["final"].items()}}


def apply_protocol(name: str) -> dict:
    """Configure the sweep for one of PROTOCOLS. Mutates ROW_CAP_BY_MODEL and
    RandomForest's parameter grid IN PLACE -- do not rebind either name, or
    callers that imported them will keep the old objects. Returns the active
    protocol dict, whose "cap_sampling" the caller must honour."""
    if name not in PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}; choose from {sorted(PROTOCOLS)}")
    spec = PROTOCOLS[name]

    ROW_CAP_BY_MODEL.clear()
    ROW_CAP_BY_MODEL.update(spec["row_caps"])

    rf_grid = MODELS["RandomForest"][1]
    rf_grid["max_depth"] = list(spec["rf_max_depth"])

    ACTIVE_PROTOCOL.clear()
    ACTIVE_PROTOCOL.update({"name": name, **spec})
    return ACTIVE_PROTOCOL


# Not part of the classical MODELS sweep -- candidate for Phase 4's exploratory
# "best classical + lightest deep learning" hybrid step.
LIGHTEST_DL_CANDIDATE = (
    "MLP",
    Pipeline([("scale", StandardScaler()),
              ("mlp", MLPClassifier(hidden_layer_sizes=(16,), max_iter=300,
                                     random_state=RANDOM_STATE))]),
)

IMBALANCE_STRATEGIES = ["none", "class_weight", "SMOTE", "ADASYN"]

# Which pipeline step (if a Pipeline) or the bare estimator carries the
# class_weight param, per model -- used only for the "class_weight" strategy.
# kNN has no native class_weight equivalent: that combination is skipped and
# recorded as not_applicable, not faked with a workaround.
_CLASS_WEIGHT_PARAM = {
    "LogisticRegression": "lr__class_weight",
    "DecisionTree": "class_weight",
    "SVM_RBF": "svc__class_weight",
    "RandomForest": "class_weight",
    "LightGBM": "class_weight",
}


def _prepare_estimator(model_name: str, imbalance: str, y_train: np.ndarray):
    """Return a fresh clone of the model, configured for `imbalance` if it's
    "class_weight" (no-op for "none"/"SMOTE"/"ADASYN", which are handled by
    resampling the training fold instead -- see nested_cv_evaluate). Returns
    None if this (model, imbalance) combination has no faithful implementation
    (kNN + class_weight) -- callers must skip and record that, not substitute
    something else silently."""
    model, _ = MODELS[model_name]
    m = clone(model)
    if imbalance != "class_weight":
        return m
    if model_name in _CLASS_WEIGHT_PARAM:
        m.set_params(**{_CLASS_WEIGHT_PARAM[model_name]: "balanced"})
        return m
    if model_name == "XGBoost":
        counts = np.bincount(y_train, minlength=2)
        neg, pos = counts[0], counts[1]
        m.set_params(scale_pos_weight=(neg / pos) if pos > 0 else 1.0)
        return m
    return None  # kNN: no native class_weight support


# The paper's central evidentiary table -- literature comparison, values
# verified against primary sources. Study scope is REFIT-only, so the DL rows
# use each paper's REFIT-specific number (not an average with UK-DALE) for a
# fair, apples-to-apples row. See paper-draft-notes.md for full verification.
LITERATURE_COMPARISON = pd.DataFrame([
    {"work": "Tabatabaei et al. (2017)", "approach": "Classical ML (RAkEL/MLkNN)",
     "dataset": "REDD", "freq": "LF", "metric": "F1", "score": 0.528},
    {"work": "Wu et al. (2019)", "approach": "Classical ML (Random Forest)",
     "dataset": "BLUED", "freq": "HF", "metric": "F1", "score": 0.98},
    {"work": "Rehmani et al. (2021)", "approach": "Classical ML (kNN/RF/ET/...)",
     "dataset": "REDD", "freq": "LF", "metric": "Accuracy (not F1 -- not directly comparable)",
     "score": 0.98},
    {"work": "Singh & Majumdar (2020)", "approach": "Classical ML (SRC)",
     "dataset": "REDD / Pecan Street", "freq": "LF", "metric": "F1", "score": 0.705},
    {"work": "Tanoni et al. (2022)", "approach": "Deep learning (CRNN)",
     "dataset": "REFIT", "freq": "LF", "metric": "weighted F1", "score": 0.83},
    {"work": "Langevin et al. (2022)", "approach": "Deep learning (CRNN-VAE)",
     "dataset": "REFIT", "freq": "LF", "metric": "weighted F1", "score": 0.78},
    {"work": "Pirnat et al. (2024)", "approach": "Deep learning (CtRNN)",
     "dataset": "REFIT", "freq": "LF", "metric": "weighted F1", "score": 0.91},
    {"work": "THIS STUDY (target)", "approach": "Classical ML (tuned, best of sweep)",
     "dataset": "REFIT", "freq": "LF", "metric": "weighted F1", "score": np.nan},
])


def nested_cv_evaluate(X: pd.DataFrame, y: np.ndarray, model_name: str,
                        imbalance: str = "none", outer_folds: int = 5) -> dict:
    """Outer CV for an honest generalisation estimate; inner grid search for
    tuning. Returns per-fold F1/precision/recall so mean +/- std and
    significance testing (Wilcoxon across folds) can be reported per the
    evaluation protocol. Returns status="not_applicable" (no folds run) for
    (model, imbalance) combinations with no faithful implementation."""
    _, param_grid = MODELS[model_name]
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold_i, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        base_model = _prepare_estimator(model_name, imbalance, y_train)
        if base_model is None:
            return {"model": model_name, "imbalance": imbalance, "status": "not_applicable",
                    "f1_mean": np.nan, "f1_std": np.nan, "precision_mean": np.nan,
                    "recall_mean": np.nan, "fold_detail": pd.DataFrame()}

        # n_jobs capped (not -1) as of the uncapped re-run (2026-09-17): with
        # ROW_CAP_BY_MODEL emptied, GridSearchCV's process-based parallelism (one
        # worker process per core, each holding its own copy of the fold's full
        # training data) started hitting numpy ArrayMemoryError on the 8GB Windows
        # machine partway through RandomForest/fridge -- n_jobs=-1 was spawning too
        # many simultaneous data copies for the now much larger uncapped folds.
        # Originally set to n_jobs=2 there; even that plus the max_depth fix below
        # still hit occasional MemoryError on the 8GB machine (system-wide memory
        # pressure, not just this process). Bumped to n_jobs=3 on 2026-09-17 once
        # moved to the MacBook (M3 Pro, 18GB RAM) -- more than double the headroom,
        # so a bit more parallelism should be safe, but this is still a
        # conservative choice, not derived from a memory calculation. This value is
        # SHARED via the synced Drive folder across both machines: if this file is
        # ever run on the 8GB Windows machine again, drop n_jobs back down to 1-2
        # there first. Like the earlier n_jobs=2 change, this only trades
        # wall-clock time for memory headroom; it does not change which
        # folds/params are searched or how scores are computed, so it has no
        # effect on the statistical validity of the uncapped results.
        if imbalance in ("SMOTE", "ADASYN"):
            sampler = SMOTE(random_state=RANDOM_STATE) if imbalance == "SMOTE" \
                else ADASYN(random_state=RANDOM_STATE)
            pipe = ImbPipeline([("sampler", sampler), ("model", base_model)])
            search = GridSearchCV(pipe, {f"model__{k}": v for k, v in param_grid.items()},
                                   scoring="f1", cv=3, n_jobs=3)
        else:
            search = GridSearchCV(base_model, param_grid, scoring="f1", cv=3, n_jobs=3)

        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)

        # Exact confusion-matrix counts (not derived from precision/recall) --
        # summed across outer folds by the caller, this is what
        # fig_confusion_matrices in make_figures.py plots per appliance.
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        fold_scores.append({
            "fold": fold_i,
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "support_positive": int(y_test.sum()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "best_params": str(search.best_params_),
        })

    scores_df = pd.DataFrame(fold_scores)
    return {
        "model": model_name, "imbalance": imbalance, "status": "ok",
        "f1_mean": scores_df["f1"].mean(), "f1_std": scores_df["f1"].std(),
        "precision_mean": scores_df["precision"].mean(),
        "recall_mean": scores_df["recall"].mean(),
        "fold_detail": scores_df,
    }


if __name__ == "__main__":
    print("Literature comparison table:")
    print(LITERATURE_COMPARISON.to_string(index=False))

    print("\nSelf-test on synthetic data (not real REFIT data)...")
    rng = np.random.default_rng(0)
    n = 2000
    Xs = pd.DataFrame({"power": rng.normal(100, 50, n), "roll_mean": rng.normal(100, 40, n)})
    ys = (rng.random(n) < 0.05).astype(int)  # ~5% positive, like a real appliance
    for m in ["LogisticRegression", "DecisionTree", "kNN"]:
        for imb in ["none", "class_weight"]:
            r = nested_cv_evaluate(Xs, ys, m, imb, outer_folds=3)
            print(f"  {m:20s} {imb:14s} -> status={r['status']:14s} f1_mean={r['f1_mean']}")
    print("Self-test complete.")
