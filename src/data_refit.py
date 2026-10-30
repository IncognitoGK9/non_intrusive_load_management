"""
REFIT data loader for the classical-ML NILM ON/OFF classification study.

REFIT (Murray, Stankovic & Stankovic, Scientific Data 2017; CC BY 4.0) provides,
per house, a CSV with columns:
    Time, Unix, Aggregate, Appliance1, Appliance2, ..., Appliance9, Issues
sampled every ~6-8 seconds. CONFIRMED against the real files (2026-09-15):
20 houses (House1-13, 15-21; House14 does not exist), row counts ~4.4M-7.4M
per house, ~6.5GB uncompressed total.

There is NO fixed "Appliance3 = fridge" mapping across houses -- it differs per
house. The original version of this module tried to infer that mapping with
fuzzy name-matching (CANONICAL_APPLIANCE_ALIASES) against a metadata file we
did not yet have. That approach is now REPLACED: HOUSE_APPLIANCE_MAP (in
house_appliance_map.py) is transcribed directly, by hand, from REFIT's own
REFIT_Readme.txt "APPLIANCE LIST" section (the primary source), so it is exact
rather than inferred. The old alias approach would in fact have mismatched --
e.g. its "fridge" aliases included "freezer", which would have wrongly tagged
House 1's separate "Chest Freezer" and "Upright Freezer" channels as fridge
readings too.

Key data-quality findings from the real files (see tables/data_audit_summary.csv
for the full per-house/appliance breakdown):
  - Not every house has every canonical appliance. House 6 and House 13 have NO
    fridge channel at all. Several houses lack a monitored microwave, kettle,
    or dishwasher (see HOUSE_APPLIANCE_MAP -- None means "not monitored").
    Per-appliance models must therefore be trained/evaluated only on the
    subset of houses that actually monitor that appliance.
  - Class imbalance is severe and appliance-dependent: e.g. microwave/kettle
    "on" time is typically well under 1.5% of samples across houses, while
    fridge "on" time ranges from ~0.2% to ~52% depending on the house/fridge
    type. This is direct, real evidence for the imbalance-handling arm of the
    study, not just a literature claim.
  - House 11 fridge reading looks anomalous (mean power while "on" ~1007W,
    versus ~80-120W in every other house) -- flagged for manual inspection
    before deciding whether to exclude House 11 from the fridge classifier.
  - House 11 and House 21 aggregate readings are noted by REFIT itself as
    affected by solar panels -- flagged as a candidate exclusion/robustness
    check for the aggregate-power features.
"""
from __future__ import annotations
import glob
import os
import pandas as pd

from house_appliance_map import HOUSE_APPLIANCE_MAP, HOUSE_APPLIANCE_NOTES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "refit_raw")  # extracted CSVs (device-local scratch)

# Standard REFIT/NILM ON-power thresholds (Watts) -- widely used defaults
# (Kelly & Knottenbelt 2015 / NILMTK conventions); a documented, deliberate
# choice, not tuned per house. Revisit as a sensitivity check later if needed.
DEFAULT_ON_THRESHOLD_W = {
    "fridge": 50,
    "washing_machine": 20,
    "dishwasher": 10,
    "microwave": 200,
    "kettle": 2000,
}

CANONICAL_APPLIANCES = list(DEFAULT_ON_THRESHOLD_W.keys())

# Houses flagged for a robustness/exclusion check -- not dropped by default,
# but every reported result should note whether they were included.
FLAGGED_HOUSES = {
    11: "Fridge channel reads anomalously high when on (~1007W vs ~80-120W elsewhere); "
        "aggregate also affected by solar panels.",
    21: "Aggregate affected by solar panels. Kettle channel excluded from HOUSE_APPLIANCE_MAP "
        "(unresolved in the current appliance-map reconstruction -- see house_appliance_map.py "
        "note on House 21) pending verification against REFIT_Readme.txt.",
}


def find_house_files(raw_dir: str = RAW_DIR) -> dict[int, str]:
    """Locate the extracted per-house REFIT CSVs. Returns {house_id: path}."""
    out = {}
    for path in glob.glob(os.path.join(raw_dir, "CLEAN_House*.csv")):
        base = os.path.basename(path)
        digits = "".join(ch for ch in base if ch.isdigit())
        if digits:
            out[int(digits)] = path
    return dict(sorted(out.items()))


def houses_with_appliance(appliance: str) -> list[int]:
    """Houses where the given canonical appliance was actually monitored."""
    return [h for h, m in HOUSE_APPLIANCE_MAP.items() if m.get(appliance) is not None]


def load_house(house_id: int, appliance: str, raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load Aggregate + the one appliance channel for a house, as a datetime-indexed frame.
    Raises KeyError if that house doesn't monitor the requested appliance."""
    ch = HOUSE_APPLIANCE_MAP[house_id].get(appliance)
    if ch is None:
        raise KeyError(f"House {house_id} does not monitor '{appliance}' (see HOUSE_APPLIANCE_MAP).")
    path = os.path.join(raw_dir, f"CLEAN_House{house_id}.csv")
    col = f"Appliance{ch}"
    df = pd.read_csv(path, usecols=["Time", "Aggregate", col, "Issues"],
                      dtype={"Aggregate": "int32", col: "int32", "Issues": "int8"},
                      parse_dates=["Time"])
    df = df.rename(columns={col: appliance}).set_index("Time")
    return df


def binarize_appliance(power_series: pd.Series, appliance: str,
                        threshold_w: float | None = None) -> pd.Series:
    """ON/OFF ground truth from raw power reading -- the real, physically
    measured label (per-appliance metering), not an artificially constructed one."""
    thr = threshold_w if threshold_w is not None else DEFAULT_ON_THRESHOLD_W.get(appliance, 15)
    return (power_series.fillna(0) >= thr).astype(int)


if __name__ == "__main__":
    files = find_house_files()
    print(f"House files found on this device: {len(files)}")
    for appliance in CANONICAL_APPLIANCES:
        hh = houses_with_appliance(appliance)
        print(f"  {appliance:16s}: monitored in {len(hh)} houses -> {hh}")
    print("\nFlagged houses (robustness check candidates):")
    for h, note in FLAGGED_HOUSES.items():
        print(f"  House {h}: {note}")
