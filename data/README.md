# Data (not included)

This folder is intentionally empty in the repository. The REFIT dataset is not redistributed here.

1. Download `CLEAN_REFIT_081116.7z` (REFIT Electrical Load Measurements, cleaned; Murray, Stankovic and Stankovic, Scientific Data 4 (2017) 160122; CC BY 4.0) from
   https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/
2. Extract `CLEAN_House1.csv` to `CLEAN_House21.csv` (no House 14) into a folder named `refit_raw` next to this repository, or anywhere and pass `--raw-dir` to `src/build_features.py`.
3. `src/build_features.py` writes the generated feature tables to `data/features/` (about 1 GB, ignored by git).

Optionally, place the archive itself at `data/CLEAN_REFIT_081116.7z`; the CNN scripts extract it once to `~/nilm_dl_cache/refit_raw` if no extracted copy is found. See Section 3 of the main README.
