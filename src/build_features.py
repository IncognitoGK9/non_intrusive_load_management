"""
PHASE 1 -- run this yourself on your machine (not in a cloud sandbox).

Builds pooled, per-appliance feature tables (Arm A and Arm B) for the
classical-ML NILM sweep, from the real REFIT house CSVs, and writes them to
data/features/<appliance>_arm<A|B>.csv (plain CSV -- deliberately not
parquet, so this step needs no extra dependency beyond pandas/numpy, which
you already have from build_features.py's own imports).

WHERE THE RAW DATA MUST BE: this script expects the extracted REFIT house
CSVs (CLEAN_House*.csv) to already exist in a folder given by --raw-dir
(default: ../../refit_raw relative to this file, i.e. a sibling of the
Drive-synced project folder -- deliberately OUTSIDE Drive, since it's ~6.5GB
raw and doesn't need to be synced). If you haven't extracted the .7z yet:

    pip install py7zr
    python3 -c "import py7zr; py7zr.SevenZipFile('data/CLEAN_REFIT_081116.7z').extractall('../../refit_raw')"

(adjust the path to wherever you want the raw CSVs to live -- just point
--raw-dir at it.)

USAGE
    python3 build_features.py                       # all 5 appliances
    python3 build_features.py --appliance fridge     # just one (useful to test first)
    python3 build_features.py --appliance fridge --houses 1,2,3   # even smaller, for a smoke test

WHAT IT DOES, per appliance:
  1. For every house that monitors that appliance (see house_appliance_map.py,
     transcribed by hand from REFIT's own documentation), load the full-
     resolution Aggregate + appliance-channel series.
  2. Compute the ON/OFF label at full resolution (binarize_appliance).
  3. Compute Arm A and Arm B features AT FULL RESOLUTION per house -- this
     matters: rolling-window features need real temporal continuity, so this
     step is never done on a subsample.
  4. ONLY AFTER features are computed, take a stratified random subsample of
     that house's rows (default cap: 60,000 rows/house, preserving the true
     ON/OFF ratio) -- this is what keeps the pooled dataset tractable on a
     laptop. This is a stated, documented methodological choice (see the
     Methods section notes it writes into run_log.txt), not a silent shortcut.
  5. Concatenate across houses (keeping a house_id column, so later analysis
     can still do grouped / leave-house-out checks even though the main sweep
     pools rows), and write one file per appliance per arm.

Progress and every choice made are also appended to features/run_log.txt so
there's a record of exactly what went into each feature file.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_refit import HOUSE_APPLIANCE_MAP, RAW_DIR, CANONICAL_APPLIANCES, \
    binarize_appliance, houses_with_appliance  # noqa: E402
from features import build_feature_matrix  # noqa: E402

SUBSAMPLE_CAP_PER_HOUSE = 60_000
RANDOM_STATE = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")


def _stratified_subsample(df: pd.DataFrame, label_col: str, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random subsample to at most `cap` rows, preserving the ON/OFF ratio. If
    the frame already has <= cap rows, returned unchanged."""
    if len(df) <= cap:
        return df
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    pos_frac = len(pos) / len(df)
    n_pos = max(1, round(cap * pos_frac)) if len(pos) > 0 else 0
    n_neg = cap - n_pos
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))
    pos_s = pos.sample(n=n_pos, random_state=rng.integers(0, 2**31 - 1)) if n_pos > 0 else pos.iloc[:0]
    neg_s = neg.sample(n=n_neg, random_state=rng.integers(0, 2**31 - 1)) if n_neg > 0 else neg.iloc[:0]
    return pd.concat([pos_s, neg_s]).sort_index()


def build_appliance_dataset(appliance: str, arm: str, house_ids: list[int],
                             raw_dir: str, log) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    frames = []
    for house_id in house_ids:
        t0 = time.time()
        path = os.path.join(raw_dir, f"CLEAN_House{house_id}.csv")
        if not os.path.exists(path):
            log(f"  SKIP house {house_id}: file not found at {path}")
            continue
        ch = HOUSE_APPLIANCE_MAP[house_id][appliance]
        col = f"Appliance{ch}"
        df = pd.read_csv(path, usecols=["Time", "Aggregate", col],
                          dtype={"Aggregate": "int32", col: "int32"},
                          parse_dates=["Time"]).set_index("Time")

        label = binarize_appliance(df[col], appliance)
        feats = build_feature_matrix(df["Aggregate"], arm=arm)
        feats["label"] = label.values
        feats["house_id"] = house_id

        before = len(feats)
        feats = _stratified_subsample(feats, "label", SUBSAMPLE_CAP_PER_HOUSE, rng)
        pct_on = 100.0 * feats["label"].mean()
        log(f"  house {house_id}: {before} rows -> subsampled to {len(feats)} "
            f"(kept %on={pct_on:.2f}), {time.time()-t0:.1f}s")
        frames.append(feats)

    if not frames:
        raise RuntimeError(f"No houses produced data for appliance={appliance!r}")
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appliance", default=None, help="one of: " + ", ".join(CANONICAL_APPLIANCES))
    ap.add_argument("--houses", default=None, help="comma-separated house ids to restrict to (for a quick test)")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_log.txt")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    appliances = [args.appliance] if args.appliance else CANONICAL_APPLIANCES
    log(f"=== build_features.py starting. appliances={appliances}, "
        f"SUBSAMPLE_CAP_PER_HOUSE={SUBSAMPLE_CAP_PER_HOUSE}, raw_dir={args.raw_dir} ===")

    for appliance in appliances:
        house_ids = houses_with_appliance(appliance)
        if args.houses:
            restrict = {int(h) for h in args.houses.split(",")}
            house_ids = [h for h in house_ids if h in restrict]
        log(f"--- {appliance}: {len(house_ids)} houses -> {house_ids} ---")

        for arm in ("A", "B"):
            t0 = time.time()
            df = build_appliance_dataset(appliance, arm, house_ids, args.raw_dir, log)
            out_path = os.path.join(args.out_dir, f"{appliance}_arm{arm}.csv")
            df.to_csv(out_path, index=False)
            log(f"  Arm {arm}: wrote {len(df)} rows, {df['label'].mean()*100:.2f}% ON, "
                f"-> {out_path} ({time.time()-t0:.1f}s total)")

    log("=== build_features.py done ===")


if __name__ == "__main__":
    main()
