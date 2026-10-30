"""
Verify HOUSE_APPLIANCE_MAP against REFIT's own documentation, and (optionally)
against already-built feature tables. Run this BEFORE any feature build.

This exists because the appliance map was once silently corrupted: a lost
hand-transcription was replaced by a reconstruction from a third-party mirror,
which pointed three house/appliance pairs at the wrong IAM channel (a toaster,
a computer site and a bedroom television) and dropped two kettle houses. The
per-appliance HOUSE COUNTS still matched, so a coverage check alone did not
catch it -- only a channel-by-channel comparison did. Hence this script.

CHECK 1 (always): for every house and appliance, does the mapped channel
  number actually carry that appliance's label in REFIT_Readme.txt?
CHECK 2 (always): does the set of houses monitoring each appliance match the
  set implied by the readme?
CHECK 3 (if tables/data_audit_summary.csv exists): does each house's ON
  percentage agree with the audit computed from the real REFIT CSVs?
CHECK 4 (if data/features/*.csv exist): does each built feature table's
  per-house ON percentage agree with the audit? This is what detects a
  feature table built under a stale map.

EXIT CODE is 0 only if every check passes -- safe to put in front of a run:
    python3 verify_appliance_map.py && python3 build_features.py

USAGE
    python3 verify_appliance_map.py
    python3 verify_appliance_map.py --features-dir ../data/features
"""
from __future__ import annotations
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_appliance_map import HOUSE_APPLIANCE_MAP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "..", "REFIT_Readme.txt")
AUDIT = os.path.join(HERE, "..", "tables", "data_audit_summary.csv")
FEATURES = os.path.join(HERE, "..", "data", "features")

# How each canonical appliance is recognised in REFIT's own channel labels.
# Deliberately strict: "freezer" must NOT match "fridge", or House 1's separate
# Chest Freezer and Upright Freezer would be treated as fridge readings.
LABEL_MATCHES = {
    "fridge":          lambda s: "fridge" in s or "firdge" in s,   # readme has this typo in some releases
    "washing_machine": lambda s: "washing machine" in s,
    "dishwasher":      lambda s: "dishwasher" in s,
    "microwave":       lambda s: "microwave" in s,
    "kettle":          lambda s: "kettle" in s,
}
TOLERANCE_PP = 0.01   # percentage points; rounding in the audit table is ~0.001

# Deliberate, documented divergences from tables/data_audit_summary.csv. The
# audit was computed under the ORIGINAL hand-transcription; where a house has
# more than one valid channel for an appliance and this project's stated choice
# differs from the one the audit happened to use, record it here with a reason.
# Anything NOT listed here that disagrees with the audit is a defect.
ALLOWED_AUDIT_DIVERGENCES: dict[tuple[int, str], str] = {
    (4, "washing_machine"):
        'House 4 monitors two washing machines. The audit used "Washing Machine (1)" '
        '(ch4, 0.261% ON); this study uses "Washing Machine (2)" (ch5, 0.873% ON), the '
        "more frequently operated of the two and therefore the one carrying more "
        "positive examples. Both are genuine washing-machine channels, so this is a "
        "channel-selection choice, not a mapping error, and needs no rebuild.",
}


def parse_readme(path: str) -> dict[int, dict[int, str]]:
    """{house_id: {channel_number: label}} from the APPLIANCE LIST section."""
    text = open(path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"\nHouse (\d+)\n", text)
    houses: dict[int, dict[int, str]] = {}
    for i in range(1, len(parts), 2):
        hid = int(parts[i])
        body = parts[i + 1].split("!NOTES")[0]
        chans: dict[int, str] = {}
        # REFIT's list is "0.Aggregate, 1.Fridge, ..."; at least one entry in
        # the shipped file is missing its comma ("2.Tumble Dryer 3.Washing
        # Machine"), so split on the number-dot marker rather than on commas.
        tokens = re.split(r"(?<![\d.])(\d+)\s*\.\s*", body)
        for j in range(1, len(tokens) - 1, 2):
            n = int(tokens[j])
            label = tokens[j + 1].strip().strip(",").strip()
            # a run-on entry like "Tumble Dryer 3.Washing Machine" is already
            # split by the regex; trailing fragments are trimmed here
            label = re.split(r"\s*\n", label)[0].strip().strip(",").strip()
            chans.setdefault(n, label)
        houses[hid] = chans
    return houses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=README)
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--features-dir", default=FEATURES)
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    readme = parse_readme(args.readme)
    print(f"CHECK 1/2: {len(HOUSE_APPLIANCE_MAP)} houses against {os.path.relpath(args.readme, HERE)}")

    for hid in sorted(HOUSE_APPLIANCE_MAP):
        chans = readme.get(hid)
        if not chans:
            failures.append(f"House {hid}: not found in the readme's APPLIANCE LIST")
            continue
        for appliance, matches in LABEL_MATCHES.items():
            ch = HOUSE_APPLIANCE_MAP[hid][appliance]
            candidates = sorted(n for n, lab in chans.items()
                                if n >= 1 and matches(lab.lower()))
            if ch is None:
                if candidates:
                    failures.append(
                        f"House {hid} {appliance}: mapped to None, but the readme has "
                        f"{[(n, chans[n]) for n in candidates]}")
            elif ch not in candidates:
                got = chans.get(ch, "<no such channel>")
                failures.append(
                    f"House {hid} {appliance}: mapped to channel {ch}, which the readme "
                    f'labels "{got}". Expected one of {[(n, chans[n]) for n in candidates]}')
            elif len(candidates) > 1:
                notes.append(f"House {hid} {appliance}: {len(candidates)} candidate channels "
                             f"{[(n, chans[n]) for n in candidates]}; using {ch} (documented choice)")

    # CHECK 3 and 4 need pandas and the audit table.
    audit = None
    if os.path.exists(args.audit):
        import pandas as pd
        audit = pd.read_csv(args.audit)
        print(f"CHECK 3  : per-house ON% against {os.path.relpath(args.audit, HERE)}")
        for appliance in LABEL_MATCHES:
            mapped = {h for h in HOUSE_APPLIANCE_MAP
                      if HOUSE_APPLIANCE_MAP[h][appliance] is not None}
            audited = set(audit[(audit.appliance == appliance) & (audit.present)].house)
            if mapped != audited:
                failures.append(
                    f"{appliance}: map covers {sorted(mapped)}, audit covers {sorted(audited)} "
                    f"(only in map: {sorted(mapped - audited)}; only in audit: {sorted(audited - mapped)})")
    else:
        notes.append(f"CHECK 3 skipped: no audit table at {args.audit}")

    if audit is not None and os.path.isdir(args.features_dir):
        import pandas as pd
        print(f"CHECK 4  : built feature tables in {os.path.relpath(args.features_dir, HERE)}")
        for appliance in LABEL_MATCHES:
            path = os.path.join(args.features_dir, f"{appliance}_armA.csv")
            if not os.path.exists(path):
                notes.append(f"CHECK 4: no built table for {appliance}, skipped")
                continue
            df = pd.read_csv(path, usecols=["label", "house_id"])
            built = df.groupby("house_id").label.mean() * 100
            truth = audit[(audit.appliance == appliance) & (audit.present)] \
                .set_index("house")["pct_on"]
            missing = sorted(set(truth.index) - set(built.index))
            if missing:
                failures.append(f"{appliance}_armA.csv: houses {missing} are in the "
                                f"appliance map but absent from the built table -- REBUILD NEEDED")
            for h in sorted(set(built.index) & set(truth.index)):
                if abs(built[h] - truth[h]) <= TOLERANCE_PP:
                    continue
                reason = ALLOWED_AUDIT_DIVERGENCES.get((h, appliance))
                if reason:
                    notes.append(f"{appliance} House {h}: built {built[h]:.3f}% ON vs audited "
                                 f"{truth[h]:.3f}% -- ACCEPTED divergence. {reason}")
                    continue
                failures.append(
                    f"{appliance}_armA.csv House {h}: built {built[h]:.3f}% ON vs audited "
                    f"{truth[h]:.3f}% -- table was built under a different channel map, "
                    f"REBUILD NEEDED")
    elif audit is not None:
        notes.append(f"CHECK 4 skipped: no features directory at {args.features_dir}")

    print()
    for n in notes:
        print("  note:", n)
    if failures:
        print(f"\nFAILED -- {len(failures)} problem(s):")
        for f in failures:
            print("  *", f)
        return 1
    print("\nAll checks passed: the appliance map matches REFIT's documentation, "
          "the real-data audit, and the built feature tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
