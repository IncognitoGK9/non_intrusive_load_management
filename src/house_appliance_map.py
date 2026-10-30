"""
Per-house REFIT appliance -> channel map for the 5 canonical appliances used
in this study (fridge, washing_machine, dishwasher, microwave, kettle).

PRIMARY SOURCE. Every entry below is transcribed from the "APPLIANCE LIST"
section of REFIT's own REFIT_Readme.txt, which ships with the dataset and is
included in this repository at ../REFIT_Readme.txt. That file is the ground
truth for this mapping. Nothing here is inferred from a third-party mirror.

CORRECTION HISTORY -- read this before citing any result built on an older
copy of this file.

  2026-09-15: the original hand-transcription (made directly from
  REFIT_Readme.txt) was lost when a cloud sandbox container was recycled. It
  was replaced by a RECONSTRUCTION derived from nilmtk's REFIT metadata
  (a third-party mirror) using the offset appliance_channel = meter - 1.

  2026-09-16: data/features/*.csv and results/ were built with that
  reconstruction.

  2026-09-17: results_uncapped/ was built with that reconstruction.

  2026-09-23: the reconstruction was checked line by line against
  REFIT_Readme.txt and against tables/data_audit_summary.csv (per-house ON
  percentages computed from the real REFIT CSVs under the ORIGINAL correct
  transcription, before the loss). FIVE DEFECTS were found and are corrected
  below. They were confirmed a third time by recomputing per-house ON
  percentages directly from the built feature tables: every house matched
  data_audit_summary.csv to within 0.001 percentage points EXCEPT the ones
  listed here.

    kettle, House 12: reconstruction used channel 6. Channel 6 is
      "Television Site(Bedroom)". The kettle is channel 4. Effect: with a
      2000 W ON-threshold a bedroom television essentially never registers,
      so House 12 contributed 60,000 rows at 0.002% ON where the true rate is
      0.845% -- roughly 506 positive samples replaced by 1, inside a pooled
      dataset that is only ~0.6% positive overall.

    kettle, House 15: reconstruction had None (not monitored). The kettle is
      channel 8. REFIT_Readme.txt renders it as "8.Kettle ," with a stray
      space, which is the likely reason the mirror lost it. True rate 0.401%.

    kettle, House 21: reconstruction had None. The kettle is channel 7,
      labelled "Kettle/Toaster" -- a COMBINED channel, so its ground truth is
      noisier than a single-appliance channel. It is included here and that
      caveat is reported, rather than silently dropping the house. True rate
      0.125%.

    microwave, House 12: reconstruction used channel 5. Channel 5 is
      "Toaster". The microwave is channel 3. A toaster also draws over the
      200 W ON-threshold, which is why this produced a plausible-looking
      0.127% instead of an obviously broken zero -- the true rate is 0.270%.

    microwave, House 13: reconstruction used channel 7. Channel 7 is
      "Computer Site". The microwave is channel 8. Built rate 0.240% against
      a true rate of 0.254%.

  Net effect: fridge (18 houses) and dishwasher (15 houses) were unaffected
  and need no rebuild. washing_machine was unaffected in channel terms (see
  the House 4 note below). microwave and kettle must be rebuilt and re-run.

A machine-checkable version of that audit ships as src/verify_appliance_map.py
and should be run before any feature build.

DOCUMENTED CHANNEL CHOICES. Where a house monitors more than one candidate
channel for an appliance, the choice is recorded here rather than resolved
silently:

  House 4: two washing machines, "Washing Machine (1)" (ch4) and "Washing
    Machine (2)" (ch5). ch5 is used, because it is the more frequently
    operated of the two (0.873% ON versus 0.261%), so it carries far more
    positive examples. ch4 is excluded. (An earlier version of this file
    described this as "ch5 used, ch6 excluded"; ch6 is the Computer Site and
    was never a candidate. The selection itself was and remains ch5.)
    Two fridge-type channels also exist here, "Fridge" (ch1) and
    "Fridge-Freezer" (ch3); the plain Fridge is used.

  House 9: a "Washer Dryer" (ch2) is excluded as a combined appliance; the
    plain "Washing Machine" (ch3) is used.

  House 10: ch1 was relabelled mid-study, from Fridge to "Magimix (Blender)"
    on 17/06/2014, per REFIT's own notes. It is NOT used. "Fridge-Freezer"
    (ch4) is used instead. No kettle channel exists in this house.

  House 11: two fridge-type channels, "Fridge" (ch1) and "Fridge-Freezer"
    (ch2); ch1 is used. FLAGGED in data_refit.py: this channel reads ~1007 W
    when on, against ~80-120 W in every other house, and REFIT notes the
    aggregate here is affected by solar panels.

  House 13: ONE microwave channel only, ch8. (An earlier version of this file
    claimed two microwave channels at ch7 and ch8 and selected ch7; ch7 is
    the Computer Site. Corrected above.)

  House 16: only fridge-freezer variants exist, no plain Fridge; the lower
    channel (ch1) is used and ch2 excluded. No microwave, no kettle.

  House 18: "Fridge(garage)" (ch1) is used over "Fridge-Freezer" (ch3) under
    the plain-Fridge-preferred rule. Note this is a garage unit rather than
    the main kitchen fridge -- a robustness/exclusion candidate alongside
    House 11.

  House 21: FLAGGED in data_refit.py -- REFIT notes the aggregate is affected
    by solar panels. Its kettle channel is the combined "Kettle/Toaster"
    noted above.

  Houses 1, 10, 16, 18: no kettle channel. Houses 1, 7, 16, 21: no microwave
  channel. House 12: no washing machine or dishwasher. Houses 6, 13: no
  fridge. These are genuine gaps in REFIT's monitoring, not omissions here.
"""
from __future__ import annotations

# channel numbers refer to the CSV columns Appliance1..Appliance9; Appliance0
# does not exist -- column 0 in REFIT's readme is the whole-house Aggregate.
HOUSE_APPLIANCE_MAP: dict[int, dict[str, int | None]] = {
    1:  {"fridge": 1,    "washing_machine": 5,    "dishwasher": 6,    "microwave": None, "kettle": None},
    2:  {"fridge": 1,    "washing_machine": 2,    "dishwasher": 3,    "microwave": 5,    "kettle": 8},
    3:  {"fridge": 2,    "washing_machine": 6,    "dishwasher": 5,    "microwave": 8,    "kettle": 9},
    4:  {"fridge": 1,    "washing_machine": 5,    "dishwasher": None, "microwave": 8,    "kettle": 9},
    5:  {"fridge": 1,    "washing_machine": 3,    "dishwasher": 4,    "microwave": 7,    "kettle": 8},
    6:  {"fridge": None, "washing_machine": 2,    "dishwasher": 3,    "microwave": 6,    "kettle": 7},
    7:  {"fridge": 1,    "washing_machine": 5,    "dishwasher": 6,    "microwave": None, "kettle": 9},
    8:  {"fridge": 1,    "washing_machine": 4,    "dishwasher": None, "microwave": 8,    "kettle": 9},
    9:  {"fridge": 1,    "washing_machine": 3,    "dishwasher": 4,    "microwave": 6,    "kettle": 7},
    10: {"fridge": 4,    "washing_machine": 5,    "dishwasher": 6,    "microwave": 8,    "kettle": None},
    11: {"fridge": 1,    "washing_machine": 3,    "dishwasher": 4,    "microwave": 6,    "kettle": 7},
    # House 12 -- CORRECTED 2026-09-23: microwave 5 -> 3 (ch5 is the Toaster),
    # kettle 6 -> 4 (ch6 is Television Site(Bedroom)).
    12: {"fridge": 1,    "washing_machine": None, "dishwasher": None, "microwave": 3,    "kettle": 4},
    # House 13 -- CORRECTED 2026-09-23: microwave 7 -> 8 (ch7 is the Computer Site).
    13: {"fridge": None, "washing_machine": 3,    "dishwasher": 4,    "microwave": 8,    "kettle": 9},
    # House 15 -- CORRECTED 2026-09-23: kettle None -> 8.
    15: {"fridge": 1,    "washing_machine": 3,    "dishwasher": 4,    "microwave": 7,    "kettle": 8},
    16: {"fridge": 1,    "washing_machine": 5,    "dishwasher": 6,    "microwave": None, "kettle": None},
    17: {"fridge": 2,    "washing_machine": 4,    "dishwasher": None, "microwave": 7,    "kettle": 8},
    18: {"fridge": 1,    "washing_machine": 5,    "dishwasher": 6,    "microwave": 9,    "kettle": None},
    19: {"fridge": 1,    "washing_machine": 2,    "dishwasher": None, "microwave": 4,    "kettle": 5},
    20: {"fridge": 1,    "washing_machine": 4,    "dishwasher": 5,    "microwave": 8,    "kettle": 9},
    # House 21 -- CORRECTED 2026-09-23: kettle None -> 7 (combined "Kettle/Toaster").
    21: {"fridge": 1,    "washing_machine": 3,    "dishwasher": 4,    "microwave": None, "kettle": 7},
}

# Channels that are COMBINED appliances rather than a single appliance. Their
# ON/OFF ground truth is noisier by construction; report this, do not hide it.
COMBINED_CHANNELS: dict[tuple[int, str], str] = {
    (21, "kettle"): 'REFIT labels House 21 channel 7 "Kettle/Toaster" -- one IAM '
                    'covering two appliances, so some labelled-ON samples are toaster '
                    'events rather than kettle events.',
    (17, "kettle"): 'REFIT notes House 17 channel 8 as "Kettle" plus "Toaster & Misc '
                    '(Occasionally plugged in)" -- predominantly kettle, with '
                    'occasional other loads on the same IAM.',
}

HOUSE_APPLIANCE_NOTES: dict[int, str] = {
    4:  "Two washing machines: Washing Machine (1) ch4 and Washing Machine (2) ch5. "
        "ch5 used (the more frequently operated of the two, 0.873% vs 0.261% ON); "
        "ch4 excluded. Two fridge-type channels (Fridge ch1, Fridge-Freezer ch3): "
        "plain Fridge (ch1) used.",
    9:  "Washer Dryer (ch2) excluded from washing_machine as a combined appliance; "
        "plain Washing Machine (ch3) used instead.",
    10: "ch1 relabelled mid-study from Fridge to Magimix (Blender) on 17/06/2014 per "
        "REFIT's own notes; NOT used. Fridge-Freezer (ch4) used instead. No kettle "
        "channel in this house.",
    11: "FLAGGED (see data_refit.py FLAGGED_HOUSES): fridge channel (ch1, plain "
        "'Fridge') reads ~1007W when on, versus ~80-120W elsewhere. A Fridge-Freezer "
        "alternative exists at ch2 but is not used. REFIT also notes the aggregate "
        "here is affected by solar panels.",
    12: "No washing_machine or dishwasher channel monitored. Microwave is ch3 and "
        "kettle is ch4 -- both were mis-mapped (to ch5 Toaster and ch6 Television "
        "Site respectively) in the 2026-09-15 reconstruction and corrected 2026-09-23.",
    13: "No fridge channel at all. ONE microwave channel, ch8 -- mis-mapped to ch7 "
        "(Computer Site) in the 2026-09-15 reconstruction and corrected 2026-09-23.",
    15: "Kettle is ch8. REFIT_Readme.txt renders this entry as '8.Kettle ,' with a "
        "stray space; the 2026-09-15 reconstruction dropped the house from the kettle "
        "set entirely. Corrected 2026-09-23.",
    16: "Only fridge-freezer variants exist (no plain Fridge): lower channel (ch1) "
        "used, ch2 excluded. No microwave or kettle channel.",
    17: "Kettle channel (ch8) carries an occasional toaster and other miscellaneous "
        "loads per REFIT's own notes -- see COMBINED_CHANNELS.",
    18: "Fridge(garage) (ch1, a secondary/garage unit) used over Fridge-Freezer (ch3) "
        "under the plain-Fridge-preferred rule -- a robustness/exclusion candidate "
        "given it is not the main kitchen fridge.",
    21: "FLAGGED (see data_refit.py FLAGGED_HOUSES): REFIT notes the aggregate is "
        "affected by solar panels. Kettle is ch7, labelled 'Kettle/Toaster' -- a "
        "combined channel (see COMBINED_CHANNELS). The 2026-09-15 reconstruction "
        "dropped this house from the kettle set; restored 2026-09-23.",
}
