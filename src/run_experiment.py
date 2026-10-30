"""
PHASE 2 -- run this yourself on your machine, after build_features.py has
produced data/features/<appliance>_arm<A|B>.csv for the appliances you want.

Runs the full classical-ML sweep: for every (appliance, feature arm, model,
imbalance-handling strategy) combination, calls nested_cv_evaluate from
model_pipeline.py (outer 5-fold CV for an honest generalisation estimate,
inner grid search for tuning) and writes:

  results/fold_results.csv     -- one row per (combo, outer fold): the raw
                                   numbers everything else is computed from.
  results/summary_results.csv  -- one row per combo: mean/std F1, precision,
                                   recall across folds, plus status (ok /
                                   not_applicable) and whether that combo was
                                   row-capped (see ROW_CAP_BY_MODEL below).

Both files are appended to incrementally, one combo at a time, and flushed
to disk after every combo -- if this script is interrupted (or you kill it
partway through an overnight run), whatever finished is already saved, and
re-running with --resume skips combos already present in fold_results.csv
rather than redoing them.

WHY THIS TAKES A WHILE: this is a genuinely large grid -- up to
5 appliances x 2 arms x 7 models x 4 imbalance strategies = 280 combinations
(fewer in practice: kNN+class_weight is not_applicable and is skipped
instantly, not fit). SVM_RBF and the tree ensembles (RandomForest/XGBoost/
LightGBM) are the slow part on REFIT's real pooled row counts (hundreds of
thousands of rows per appliance) -- each gets its own row cap
(model_pipeline.ROW_CAP_BY_MODEL), applied per combo and logged, so training
runs on a bounded random subsample of that fold's data rather than the full
pooled dataset. This is a documented, stated methodological choice (see
run_log.txt and the "row-capped" note next to every affected combo), not a
silent shortcut -- and the caps were added/widened on 2026-09-16 after a
real run on REFIT data showed the uncapped grid would take days across all
5 appliances (see model_pipeline.py's comment on ROW_CAP_BY_MODEL for the
measured numbers this was based on). Run --appliance / --model / --imbalance
flags let you split the grid into smaller chunks (e.g. run kettle overnight,
fridge the next night), or run just one model first to sanity-check timing
on your machine before committing to the full sweep.

UPDATE (2026-09-17): after reviewing the capped results, ROW_CAP_BY_MODEL in
model_pipeline.py was emptied ({}) -- every model now trains uncapped, on the
full pooled dataset, for a follow-up run. The original capped run's output is
untouched in results/ (row_capped=True rows there record what was actually
run the first time -- see ROW_CAP_BY_MODEL_HISTORICAL_FIRST_PASS in
model_pipeline.py for those cap sizes). Point --out-dir at a fresh directory
(e.g. results_uncapped/) for the uncapped re-run so the two are never mixed,
and use --model to redo only the four models that were actually capped
(SVM_RBF, RandomForest, XGBoost, LightGBM) -- Logistic Regression, Decision
Tree, and k-NN were already uncapped the first time and don't need re-running.

REQUIRED PACKAGES beyond pandas/numpy/scikit-learn (install with pip on your
own machine -- this is exactly the dependency set model_pipeline.py imports):
    pip install xgboost lightgbm imbalanced-learn

USAGE
    python3 run_experiment.py                                   # full grid, all appliances/arms/models/imbalance strategies
    python3 run_experiment.py --appliance kettle                # just one appliance (both arms, all models)
    python3 run_experiment.py --appliance kettle --arm B         # one appliance, one arm
    python3 run_experiment.py --model LogisticRegression,DecisionTree   # a couple of fast models first, as a sanity check
    python3 run_experiment.py --resume                          # continue an interrupted run, skipping completed combos
    python3 run_experiment.py --model RandomForest --out-dir ../results_uncapped --resume   # uncapped re-run, one model, separate output dir
"""
from __future__ import annotations
import os

# Silence noisy library warnings that don't affect correctness (e.g. sklearn's
# FutureWarning about _validate_data being renamed in a future release) -- they
# just clutter run_log.txt / the terminal during long unattended sweeps. Real
# failures still raise exceptions and are not affected by this. This must be
# set via the environment (not just warnings.filterwarnings below) BEFORE
# sklearn/joblib are imported, because GridSearchCV(n_jobs=-1) spawns separate
# worker processes -- each one reads PYTHONWARNINGS at its own startup, but
# never sees a warnings.filterwarnings() call made only in this process.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::DeprecationWarning")

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from data_refit import CANONICAL_APPLIANCES  # noqa: E402
from model_pipeline import MODELS, IMBALANCE_STRATEGIES, ROW_CAP_BY_MODEL, \
    PROTOCOLS, apply_protocol, nested_cv_evaluate  # noqa: E402

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RANDOM_STATE = 42


def _combo_key(appliance: str, arm: str, model: str, imbalance: str) -> tuple:
    return (appliance, arm, model, imbalance)


def _load_done_combos(summary_results_path: str) -> set[tuple]:
    """Read completed combos from summary_results.csv, not fold_results.csv --
    a not_applicable or skipped_too_slow combo writes no fold rows at all, so
    fold_results.csv alone would make --resume re-run those forever."""
    if not os.path.exists(summary_results_path):
        return set()
    df = pd.read_csv(summary_results_path, usecols=["appliance", "arm", "model", "imbalance"])
    return set(df.drop_duplicates().itertuples(index=False, name=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appliance", default=None,
                    help="comma-separated subset of: " + ", ".join(CANONICAL_APPLIANCES))
    ap.add_argument("--arm", default=None, help="comma-separated subset of: A, B")
    ap.add_argument("--model", default=None,
                    help="comma-separated subset of: " + ", ".join(MODELS.keys()))
    ap.add_argument("--imbalance", default=None,
                    help="comma-separated subset of: " + ", ".join(IMBALANCE_STRATEGIES))
    ap.add_argument("--features-dir", default=FEATURES_DIR)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="skip combos already present in fold_results.csv")
    ap.add_argument("--protocol", default="final", choices=sorted(PROTOCOLS),
                    help="which published protocol to run: 'first_pass' (per-model row "
                         "caps, pure random subsampling, RandomForest tuned over "
                         "max_depth in {None, 12}); 'final' (tree ensembles uncapped, "
                         "SVM_RBF capped at 100k with stratified subsampling, "
                         "RandomForest tuned over max_depth=12 only); or "
                         "'depth_control' (first_pass in every respect except "
                         "RandomForest's max_depth=12 only, which isolates the effect "
                         "of the depth restriction from the effect of the data "
                         "increase). Default: final.")
    args = ap.parse_args()

    protocol = apply_protocol(args.protocol)

    appliances = args.appliance.split(",") if args.appliance else CANONICAL_APPLIANCES
    arms = args.arm.split(",") if args.arm else ["A", "B"]
    models = args.model.split(",") if args.model else list(MODELS.keys())
    imbalances = args.imbalance.split(",") if args.imbalance else IMBALANCE_STRATEGIES

    for name, valid in [("model", MODELS.keys()), ("imbalance", IMBALANCE_STRATEGIES)]:
        pass  # validated implicitly below by KeyError if bad name given

    os.makedirs(args.out_dir, exist_ok=True)
    fold_path = os.path.join(args.out_dir, "fold_results.csv")
    summary_path = os.path.join(args.out_dir, "summary_results.csv")
    log_path = os.path.join(args.out_dir, "run_log.txt")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    done_combos = _load_done_combos(summary_path) if args.resume else set()
    if args.resume:
        log(f"--resume: {len(done_combos)} combos already completed, will be skipped")

    log(f"=== run_experiment.py starting. protocol={args.protocol!r}, "
        f"appliances={appliances}, arms={arms}, models={models}, "
        f"imbalance={imbalances}, outer_folds={args.outer_folds} ===")
    log(f"    protocol settings: row_caps={protocol['row_caps']}, "
        f"cap_sampling={protocol['cap_sampling']!r}, "
        f"RandomForest max_depth grid={MODELS['RandomForest'][1]['max_depth']}")

    fold_rows_buffer = []
    summary_rows_buffer = []

    def flush():
        nonlocal fold_rows_buffer, summary_rows_buffer
        if fold_rows_buffer:
            df = pd.DataFrame(fold_rows_buffer)
            df.to_csv(fold_path, mode="a", header=not os.path.exists(fold_path), index=False)
            fold_rows_buffer = []
        if summary_rows_buffer:
            df = pd.DataFrame(summary_rows_buffer)
            df.to_csv(summary_path, mode="a", header=not os.path.exists(summary_path), index=False)
            summary_rows_buffer = []

    n_total = len(appliances) * len(arms) * len(models) * len(imbalances)
    n_done = 0

    for appliance in appliances:
        for arm in arms:
            feat_path = os.path.join(args.features_dir, f"{appliance}_arm{arm}.csv")
            if not os.path.exists(feat_path):
                log(f"SKIP {appliance} arm{arm}: feature file not found at {feat_path} "
                    f"(run build_features.py --appliance {appliance} first)")
                n_done += len(models) * len(imbalances)
                continue

            df = pd.read_csv(feat_path)
            X_full = df.drop(columns=["label", "house_id"])
            y_full = df["label"].values
            n_rows = len(df)

            for model_name in models:
                for imbalance in imbalances:
                    n_done += 1
                    key = _combo_key(appliance, arm, model_name, imbalance)
                    if key in done_combos:
                        log(f"[{n_done}/{n_total}] SKIP (resume) {key}")
                        continue

                    t0 = time.time()
                    row_cap = ROW_CAP_BY_MODEL.get(model_name)
                    if row_cap is not None and n_rows > row_cap:
                        # STRATIFIED subsampling (changed 2026-09-19, session 7/8) --
                        # the original historical-first-pass cap used plain
                        # rng.choice() (pure random, no regard for class balance).
                        # Switched to stratified sampling here so the capped
                        # subsample preserves the true ON/OFF class proportion --
                        # important for rare-event appliances (microwave 0.27% ON,
                        # kettle 0.71% ON) where a pure random draw could, by chance,
                        # meaningfully skew the effective imbalance the model is
                        # tuned/evaluated against relative to the full dataset.
                        # train_test_split(..., stratify=y_full) achieves this;
                        # `idx` in place of the actual arrays for the same
                        # deterministic-seed behaviour as before.
                        # Which of the two is used is set by --protocol, not by
                        # editing this file: the first pass used pure random
                        # subsampling, the final run used stratified. Keeping both
                        # reachable is what makes each published run reproducible.
                        if protocol["cap_sampling"] == "stratified":
                            idx_all = np.arange(n_rows)
                            idx, _ = train_test_split(idx_all, train_size=row_cap,
                                                       stratify=y_full,
                                                       random_state=RANDOM_STATE)
                        else:
                            rng = np.random.default_rng(RANDOM_STATE)
                            idx = rng.choice(n_rows, size=row_cap, replace=False)
                        X, y = X_full.iloc[idx].reset_index(drop=True), y_full[idx]
                        capped_note = (f" (row-capped {n_rows}->{row_cap}, "
                                       f"{protocol['cap_sampling']})")
                    else:
                        X, y = X_full, y_full
                        capped_note = ""

                    result = nested_cv_evaluate(X, y, model_name, imbalance,
                                                 outer_folds=args.outer_folds)
                    elapsed = time.time() - t0

                    log(f"[{n_done}/{n_total}] {appliance:16s} arm{arm} {model_name:20s} "
                        f"{imbalance:14s} -> status={result['status']:14s} "
                        f"f1_mean={result['f1_mean']}{capped_note} ({elapsed:.1f}s)")

                    if result["status"] == "ok":
                        for _, fold_row in result["fold_detail"].iterrows():
                            fold_rows_buffer.append({
                                "appliance": appliance, "arm": arm, "model": model_name,
                                "imbalance": imbalance, "fold": fold_row["fold"],
                                "f1": fold_row["f1"], "precision": fold_row["precision"],
                                "recall": fold_row["recall"],
                                "support_positive": fold_row["support_positive"],
                                # exact confusion-matrix counts, summed across folds by
                                # make_figures.py to plot a real per-appliance confusion matrix
                                "tn": fold_row["tn"], "fp": fold_row["fp"],
                                "fn": fold_row["fn"], "tp": fold_row["tp"],
                                "best_params": fold_row["best_params"],
                            })

                    summary_rows_buffer.append({
                        "appliance": appliance, "arm": arm, "model": model_name,
                        "imbalance": imbalance, "status": result["status"],
                        "protocol": args.protocol,
                        "n_rows": n_rows, "row_capped": bool(capped_note),
                        "f1_mean": result["f1_mean"], "f1_std": result["f1_std"],
                        "precision_mean": result["precision_mean"],
                        "recall_mean": result["recall_mean"],
                        "elapsed_seconds": round(elapsed, 1),
                    })
                    flush()  # write after every combo -- never lose more than one combo's work

    log("=== run_experiment.py done ===")


if __name__ == "__main__":
    main()
