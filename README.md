# What matters in classical machine learning for non-intrusive load monitoring?

Code, configuration files and complete results for the study **"What matters in classical machine learning for non-intrusive load monitoring? A factorial benchmark on REFIT with validation on unseen houses"** (Zablon Ombiro, Strathmore University). The paper is being submitted to *Energy and Buildings*; a preprint will be posted on arXiv.

The study asks which design choices determine the performance of classical machine learning for appliance ON/OFF classification in non-intrusive load monitoring (NILM), and which of the resulting conclusions survive validation on houses that the models never saw during training.

This repository contains everything needed to reproduce every number, table and figure of the paper. It does **not** contain the REFIT dataset (download instructions in Section 3) or the manuscript.

---

## Contents

1. [Study design](#1-study-design)
2. [Main results](#2-main-results)
3. [Dataset: where to get it and where to put it](#3-dataset-where-to-get-it-and-where-to-put-it)
4. [Environment](#4-environment)
5. [Quick reproduction from the stored results (minutes)](#5-quick-reproduction-from-the-stored-results-minutes)
6. [Full reproduction from the raw data (days)](#6-full-reproduction-from-the-raw-data-days)
7. [Where each number, table and figure comes from](#7-where-each-number-table-and-figure-comes-from)
8. [Repository layout](#8-repository-layout)
9. [Result file formats](#9-result-file-formats)
10. [Computing environment and run times](#10-computing-environment-and-run-times)
11. [Protocol details and known deviations](#11-protocol-details-and-known-deviations)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations](#13-limitations)
14. [Citation and licence](#14-citation-and-licence)

---

## 1. Study design

| Factor | Levels |
|---|---|
| Appliance | fridge, washing machine, dishwasher, kettle, microwave |
| Model | logistic regression (LR), decision tree (DT), k-nearest neighbours (k-NN), support vector machine with RBF kernel (SVM-RBF), Random Forest (RF), XGBoost, LightGBM |
| Feature set | FS-A: aggregate active power and hour of day (2 features). FS-B: FS-A plus 15-sample trailing rolling mean, standard deviation, minimum, maximum, skewness and kurtosis of the aggregate power, its first difference, the rolling mean absolute difference, and day of week (11 features) |
| Imbalance strategy | none, class weighting, SMOTE, ADASYN |

7 models x 2 feature sets x 4 strategies x 5 appliances = **280 designed configurations; 270 are fitted** (k-NN has no native class weighting, so k-NN with class weighting is recorded as not applicable).

**Labels.** A sample is ON when the appliance channel reaches a fixed power threshold (W): fridge 50, washing machine 20, dishwasher 10, kettle 2000, microwave 200 (thresholds of Kelly and Knottenbelt, 2015, without minimum ON/OFF durations).

**Sampling.** Features are computed at the full 8 s resolution; each house is then reduced to 60,000 samples by stratified random sampling (preserving its ON/OFF ratio), and the houses that monitor an appliance are pooled (0.90 to 1.14 million samples per appliance).

**Metric.** F1 score of the ON class. Accuracy, precision and recall are recorded for every fold. The support-weighted F1 over the five appliances weights each appliance by its number of ON samples.

**Validation protocols.**

1. **Pooled nested cross-validation**: 5 stratified outer folds over the pooled samples of all houses; 3-fold inner grid search maximising F1.
2. **Leave-house-out validation (unseen houses)**: 5 folds by stratified group k-fold over houses (each fold tests 3 or 4 houses not seen in training). Hyperparameters are fixed at the value most often selected in the pooled nested CV; oversampling is applied to the training houses only.

**Robustness checks.** Three further random house partitions (four in total) for LR, DT, XGBoost and LightGBM; house-grouped hyperparameter tuning and nested house-grouped selection; a same-protocol one-dimensional convolutional network (CNN) on raw aggregate windows; and a data-quality sensitivity run without the solar-affected Houses 11 and 21 and the House 4 kettle channel.

## 2. Main results

| Result | Pooled CV | Unseen houses |
|---|---|---|
| Support-weighted F1 of the best pooled configurations | 0.548 | 0.447 |
| Gain from engineered features (FS-B over FS-A), mean F1 | +52% | +52% |
| Gain from class weighting over no correction (k-NN excluded) | +32% | +69% |
| Differences between models (Friedman test) | p = 0.002 | p = 0.57 |
| Share of F1 variance: appliance / feature set / model / imbalance | 37.7 / 13.2 / 7.5 / 2.1% | 33.7 / 13.1 / 3.6 / 9.1% |
| Nested house-grouped vs pooled selection (16 configurations per appliance) | | better for 3 of 5 appliances (up to +0.110); weighted F1 0.455 vs 0.446 |
| Same-protocol CNN, support-weighted F1 | | 0.450 (phase D); 0.496 causal, 0.497 centred (phase R_CNN) |

## 3. Dataset: where to get it and where to put it

**Dataset.** REFIT Electrical Load Measurements (cleaned version). D. Murray, L. Stankovic, V. Stankovic, *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*, Scientific Data 4 (2017) 160122, https://doi.org/10.1038/sdata.2016.122. Licence: CC BY 4.0. 20 houses (numbered 1 to 13 and 15 to 21), aggregate plus up to 9 appliance channels at about 8 s, about two years per house.

**Download.** University of Strathclyde data portal:
https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/

Download the archive `CLEAN_REFIT_081116.7z` (about 1 GB compressed; about 6 GB extracted). The portal also provides `CLEAN_READ_ME_081116.txt` and the metadata workbook; keep them for reference.

**Where to put it.** The scripts look for the extracted files `CLEAN_House1.csv` to `CLEAN_House21.csv` (19 + 1 files; there is no House 14). Use either of these layouts:

```
Option A (default of the feature builder): a folder next to the repository
  parent/
    non_intrusive_load_management/   (this repository)
    refit_raw/                       CLEAN_House1.csv ... CLEAN_House21.csv

Option B: any folder, passed explicitly
  python3 build_features.py --raw-dir /path/to/refit_raw
```

The CNN scripts (phase D and phase R_CNN), which read raw windows, search in this order: `../refit_raw` (next to the repository), `~/refit_raw`, `data/refit_raw`, `~/nilm_dl_cache/refit_raw`. If none exists and the archive is at `data/CLEAN_REFIT_081116.7z`, they extract it once to `~/nilm_dl_cache/refit_raw`.

**Extracting the archive.**

```bash
pip install py7zr
python3 -c "import py7zr; py7zr.SevenZipFile('CLEAN_REFIT_081116.7z').extractall('../refit_raw')"
# or, with p7zip installed:  7z x CLEAN_REFIT_081116.7z -o../refit_raw
```

**Generated feature tables.** Step 2 of Section 6 writes `data/features/<appliance>_arm<A|B>.csv` (10 files, about 1 GB). They are not stored in the repository (`.gitignore`); the classical experiments read them.

**The repository never redistributes the dataset.** `data/README.md` repeats these instructions.

## 4. Environment

All experiments were run with **Python 3.9.6 on macOS** (Apple MacBook Pro, M3 Pro, 18 GB memory). Exact versions:

| Package | Version | Package | Version |
|---|---|---|---|
| numpy | 2.0.2 | xgboost | 2.1.4 |
| pandas | 2.3.3 | lightgbm | 4.6.0 |
| scipy | 1.13.1 | torch | 2.8.0 |
| scikit-learn | 1.6.1 | matplotlib | 3.9.4 |
| imbalanced-learn | 0.12.4 | py7zr | 1.0.0 |
| statsmodels | 0.14.4 (analysis only) | openpyxl | 3.1.5 (analysis only) |

`requirements.txt` pins these versions; `requirements-lock.txt` is the full `pip freeze` of the machine that ran the experiments.

```bash
git clone https://github.com/IncognitoGK9/non_intrusive_load_management.git
cd non_intrusive_load_management
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- Python 3.9 or later works; other library versions may change results in the third decimal place. The reference house partition is stored as a file (Section 11) so that leave-house-out folds do not depend on the scikit-learn version.
- LightGBM on macOS needs OpenMP: `brew install libomp`.
- The CNN uses Apple's MPS backend when available and the CPU otherwise.
- The shell scripts use `caffeinate` to keep a Mac awake. On Linux or Windows delete `caffeinate -dimsu -w $$ &` (and `caffeinate -dimsu` in `run_lho_factorial.sh`) or replace it with `systemd-inhibit`.
- All scripts are run **from `src/`** and use paths relative to the repository root. Random seed 42 is used throughout (seeds 1, 2 and 3 for the extra house partitions).

## 5. Quick reproduction from the stored results (minutes)

All raw fold-level results are stored in the repository, so every reported number, figure and supplementary table can be regenerated **without the dataset and without retraining**:

```bash
cd src
python3 consolidate_results.py        # results_final/pooled_first_pass_280.csv, pooled_final_280.csv
python3 analyse_unseen_houses.py      # results_final/analysis_unseen_houses.json, lho270_summary.csv, anova_pooled_270.json
python3 analyse_revision.py           # results_final/revision_summary.json, nested_selection_phaseC.csv
python3 make_paper_figures.py         # figures/fig01 ... fig09, graphical_abstract.png
python3 build_supplementary.py        # results_final/Supplementary_Material.xlsx (Tables S1 to S12)
```

Expected outcome: the JSON and CSV files in `results_final/` are regenerated identically (this was verified on macOS and on Linux: 320 values in `analysis_unseen_houses.json`, 202 in `revision_summary.json` and 27 in `anova_pooled_270.json` matched exactly). Run times: under a minute in total.

Supplementary Tables S13 to S15 of the paper are taken from `results_final/nested_selection_phaseC.csv`, `results_all_experiments/phase_R_CNN/fold_results.csv` and `results_all_experiments/phase_R_SENS/fold_results.csv`.

## 6. Full reproduction from the raw data (days)

Run in this order from `src/`. Every experiment script is resumable: if it stops, run the same command again and finished configurations or folds are skipped (`--resume` for `run_experiment.py`; automatic for the others). Times are logged computing times on the machine in Section 10.

### Step 1. Verify the appliance-channel map (seconds)

```bash
python3 verify_appliance_map.py
```

Checks the per-house appliance-to-channel map in `house_appliance_map.py` against the REFIT documentation and the data audit (`tables/data_audit_summary.csv`); after Step 2 it also checks the built feature tables. Exit code 0 means every check passed.

### Step 2. Build the feature tables (1 to 2 h)

```bash
python3 build_features.py --raw-dir ../../refit_raw
# optional: --appliance kettle   --houses 2,3   (quick test on a subset)
```

Writes `data/features/<appliance>_arm<A|B>.csv` (A = FS-A, B = FS-B) and a log. Features are computed at full resolution before the stratified 60,000-sample subsampling per house.

### Step 3. Pooled nested CV, first pass (logged 11.6 h)

```bash
python3 run_experiment.py --protocol first_pass --out-dir ../results --resume
python3 compute_accuracy_supplement.py --dir ../results
```

All 7 models; SVM-RBF, RF, XGBoost and LightGBM on row-capped subsamples (30,000, 60,000, 100,000 and 100,000 rows, uniform random).

### Step 4. Pooled nested CV, final run (logged 75.5 h)

```bash
python3 run_experiment.py --protocol final --model SVM_RBF,RandomForest,XGBoost,LightGBM --out-dir ../results_uncapped --resume
python3 compute_accuracy_supplement.py --dir ../results_uncapped
```

RF, XGBoost and LightGBM on the full pooled data (RF maximum depth fixed at 12); SVM-RBF on 100,000 stratified rows. LR, DT and k-NN are not re-run; their final values are the first-pass values.

### Step 5. Corrected kettle and microwave runs

The kettle and microwave channels of Houses 12, 13, 15 and 21 were mis-mapped in the original runs of Steps 3 and 4; the map in `house_appliance_map.py` is now correct. The corrected runs are stored separately and replace the original kettle and microwave rows during consolidation:

```bash
python3 run_experiment.py --protocol first_pass --appliance kettle,microwave --out-dir ../results_fixed_first_pass --resume
python3 run_experiment.py --protocol final --appliance kettle,microwave --model SVM_RBF,RandomForest,XGBoost,LightGBM --out-dir ../results_fixed_final --resume
```

If you reproduce from scratch with the corrected map, Steps 3 and 4 already produce correct kettle and microwave rows; Step 5 is still needed if you use `consolidate_results.py` unchanged, because it reads these two folders.

### Step 6. Random Forest depth control (hours; not timed)

```bash
python3 run_experiment.py --protocol depth_control --model RandomForest --out-dir ../results_depth_control --resume
```

RF on the first-pass subsamples and seed with the depth grid restricted to 12; isolates the effect of the depth restriction from that of the training-set size (Section 5.3 of the paper).

### Step 7. Consolidate the pooled results (seconds)

```bash
python3 consolidate_results.py
```

### Step 8. Leave-house-out plans (seconds)

```bash
python3 make_lho_plans.py
```

Writes the fixed hyperparameters (modal value over the five outer folds) to `results_lho_factorial/plan_factorial.json` and `results_all_experiments/config/plan_A.json`, `plan_B.json`, `plan_E.json`. The reference house partition is stored in `results_all_experiments/config/house_partition.csv` and `results_lho_factorial/house_partition.csv` (identical).

### Step 9. Leave-house-out, best configuration of each model (35 configurations; logged 4.4 h)

Stored in `results_leave_house_out_all/` (plan in `plan.json`, partition in `house_partition.csv`). These 35 configurations are also contained in the full factorial of Steps 10 and 11; the folder is kept because Section 4.8 and Supplementary Table S6 of the paper use it.

### Step 10. Leave-house-out factorial of LR, DT and XGBoost (120 configurations, 600 folds; logged 1.6 h)

```bash
bash run_lho_factorial.sh
```

### Step 11. Leave-house-out phases A to E (logged 49 h in total)

```bash
bash run_all_experiments.sh          # runs the phases in three parallel chains: A then E, B, C then D
# or one phase at a time:
python3 run_all_experiments.py --phase A
```

| Phase | Content | Folds | Logged time |
|---|---|---|---|
| A | LightGBM, RF, k-NN factorial, reference partition (110 configurations) | 550 | 7.8 h |
| B | LR, DT, XGBoost, LightGBM factorial on three further partitions (seeds 1, 2, 3) | 2,400 | 4.8 h |
| C | House-grouped tuning and selection, FS-B, LR, DT, XGBoost, LightGBM (80 configurations) | 400 | 5.5 h |
| D | Same-protocol CNN, fixed settings (needs the raw CSV files) | 25 | 0.8 h |
| E | SVM-RBF factorial, 100,000 stratified training rows (40 configurations) | 200 | 29.6 h |

Outputs: `results_all_experiments/phase_<X>/fold_results.csv`; logs in `results_all_experiments/logs/`.

### Step 12. Revision runs: CNN variants and data-quality sensitivity (logged 2.9 h)

```bash
bash run_revision.sh
# or: python3 run_revision_experiments.py --phase CNN    and    python3 run_revision_experiments.py --phase SENS
```

- `CNN`: the phase-D network with about 20% of the training houses held out as validation houses for early stopping (at most 12 epochs, patience 3) and for the decision threshold; centred 99-sample and causal 50-sample windows (50 folds, 1.5 h; needs the raw CSV files).
- `SENS`: LR, DT, XGBoost and LightGBM factorial plus the RF, FS-B, class-weighting fridge configuration without Houses 11 and 21 and without the House 4 kettle channel (161 configurations, 805 folds, 1.3 h).

The script then runs `analyse_revision.py` and records `pip freeze` in `requirements-lock.txt`.

### Step 13. Analyses, figures and tables

Run the five commands of Section 5.

## 7. Where each number, table and figure comes from

| Paper item | Source file(s) | Produced by |
|---|---|---|
| Table 2, Table S1 (data audit) | `tables/data_audit_summary.csv` (stored per-house audit of the raw data) | read by `verify_appliance_map.py` and `build_supplementary.py` |
| Tables 6 to 8, Sections 4.1 to 4.6, Tables S2 to S4 | `results_final/pooled_first_pass_280.csv`, `pooled_final_280.csv` | `consolidate_results.py` |
| Section 5.3, Table S5 (RF depth control) | `results_depth_control/` | Step 6 |
| Table 9 | `results_final/analysis_unseen_houses.json` | `analyse_unseen_houses.py` |
| Section 4.8 (model comparison), Table S6 (35 configurations) | `results_leave_house_out_all/fold_results_unified.csv` | Step 9 |
| Section 4.7 and 4.9, Table 10, Fig. 8, Fig. 9, Tables S7 to S9 | `results_final/analysis_unseen_houses.json`, `anova_pooled_270.json`, `lho270_summary.csv` | `analyse_unseen_houses.py` |
| Section 4.10 (partitions, tuning), Tables S10, S11 | `results_all_experiments/phase_B`, `phase_C` | `analyse_unseen_houses.py` |
| Nested selection, Table S13 | `results_final/nested_selection_phaseC.csv`, `revision_summary.json` | `analyse_revision.py` |
| Table 11, Tables S12, S14 (CNN) | `phase_D`, `phase_R_CNN` | `analyse_unseen_houses.py`, `analyse_revision.py` |
| Data-quality sensitivity, Table S15 | `phase_R_SENS` | `analyse_revision.py` |
| Matched class-weighting gains, two-way ANOVA without k-NN, rank ANOVA, Iman-Davenport | `results_final/revision_summary.json` | `analyse_revision.py` |
| Figs. 1 to 9, graphical abstract | `figures/` | `make_paper_figures.py` |
| Appendix A (run times) | `elapsed_seconds` columns of all `fold_results.csv` and `summary_results.csv` files | summed directly |

## 8. Repository layout

```
README.md, LICENSE, CITATION.cff, requirements.txt, requirements-lock.txt
data/README.md                 where to download and place REFIT (no data in the repository)
src/
  data_refit.py                loading of REFIT house files, ON thresholds, labels
  house_appliance_map.py       verified per-house appliance-to-channel map (with change history)
  verify_appliance_map.py      checks the map against documentation, audit and feature tables
  features.py                  FS-A and FS-B feature definitions
  build_features.py            builds data/features/<appliance>_arm<A|B>.csv
  model_pipeline.py            models, hyperparameter grids, imbalance strategies, protocols
  run_experiment.py            pooled nested CV (--protocol first_pass | final | depth_control)
  compute_accuracy_supplement.py          adds accuracy to the summary files
  compute_capped_vs_uncapped_comparison.py  first pass vs final run per configuration
  consolidate_results.py       canonical pooled results for all 280 configurations
  make_lho_plans.py            fixed hyperparameters for the leave-house-out runs
  run_lho_factorial.py, .sh    leave-house-out factorial, LR, DT, XGBoost
  run_all_experiments.py, .sh  leave-house-out phases A to E
  run_revision_experiments.py, run_revision.sh   CNN variants and data-quality sensitivity
  revision_status.sh           progress of the revision runs
  analyse_unseen_houses.py     leave-house-out statistics and ANOVA
  analyse_revision.py          revision statistics and nested selection
  make_paper_figures.py        all figures
  build_supplementary.py       supplementary workbook (Tables S1 to S12)
results/                       pooled first pass, original runs
results_uncapped/              pooled final run of SVM-RBF, RF, XGBoost, LightGBM, original runs
results_fixed_first_pass/      corrected kettle and microwave, first pass
results_fixed_final/           corrected kettle and microwave, final run
results_depth_control/         RF depth control
results_leave_house_out_all/   35 configurations, reference partition
results_lho_factorial/         leave-house-out factorial, LR, DT, XGBoost
results_all_experiments/       phases A to E, R_CNN, R_SENS; config/ (plans, reference partition); logs/
results_final/                 consolidated results and analysis outputs
tables/data_audit_summary.csv  per-house data audit
figures/                       figures of the paper (PNG, 300 dpi; Fig. 1 also PDF)
```

## 9. Result file formats

- `fold_results.csv`: one row per configuration and outer fold. Columns include `appliance`, `arm` (A = FS-A, B = FS-B), `model`, `imbalance`, `fold`, `test_houses` (leave-house-out only), `n_train`, `n_test`, `n_test_on`, `f1`, `precision`, `recall`, `accuracy`, the confusion counts `tn`, `fp`, `fn`, `tp`, `best_params` (pooled runs) and `elapsed_seconds`.
- `summary_results.csv`: one row per configuration with fold means and standard deviations (`f1_mean`, `f1_std`, `precision_mean`, `recall_mean`), `n_rows`, `row_capped`, `status` (`ok` or `not_applicable` for k-NN with class weighting) and `elapsed_seconds`.
- `run_log.txt`, `*.log`: time-stamped progress logs of the original runs.
- `house_partition.csv`: house-to-fold assignment of the reference partition, per appliance.
- `plan*.json`: fixed hyperparameters per configuration for the leave-house-out runs.

## 10. Computing environment and run times

All experiments were run locally and sequentially on an Apple MacBook Pro with an M3 Pro processor and 18 GB of unified memory. Logged computing times (sums of per-fold or per-configuration times, including hyperparameter search):

| Experiment | Folds | Time (h) |
|---|---|---|
| Pooled nested CV, first pass (270 configurations) | 1,350 | 11.6 |
| Pooled nested CV, final run of SVM-RBF, RF, XGBoost, LightGBM (160 configurations) | 800 | 75.5 |
| Uncapped SVM-RBF attempt (stopped; no configuration completed) | 0 | about 9 |
| Leave-house-out, best configuration of each model (35 configurations) | 175 | 4.4 |
| Leave-house-out factorial, LR, DT, XGBoost (120 configurations) | 600 | 1.6 |
| Leave-house-out factorial, LightGBM, RF, k-NN (110 configurations) | 550 | 7.8 |
| Leave-house-out factorial, SVM-RBF (40 configurations) | 200 | 29.6 |
| Repeated house partitions (3 x 160 configurations) | 2,400 | 4.8 |
| House-grouped tuning and selection (80 configurations) | 400 | 5.5 |
| Same-protocol CNN, fixed settings | 25 | 0.8 |
| Same-protocol CNN, centred and causal windows | 50 | 1.5 |
| Data-quality sensitivity (161 configurations) | 805 | 1.3 |

Feature construction took 1 to 2 h; the RF depth-control run was not timed. In total about 150 h of logged computing time, and 5,205 models fitted and tested on held-out houses.

## 11. Protocol details and known deviations

- **Training-set size.** The first pass capped SVM-RBF (30,000), RF (60,000), XGBoost and LightGBM (100,000 rows, uniform random). The final run removed the caps for RF, XGBoost and LightGBM; SVM-RBF did not complete any configuration on the full data in about 9 h and was re-run on 100,000 stratified rows.
- **RF depth.** Unlimited depth on up to 1.14 million rows exceeded the 18 GB of memory, so the final RF grid fixes the maximum depth at 12. `results_depth_control/` separates this effect from the training-set effect.
- **Channel map.** Kettle (Houses 12, 15, 21) and microwave (Houses 12, 13) were mis-mapped in the original runs and corrected; see Step 5 and the history in `house_appliance_map.py`. In House 4 the more frequently used of two washing machines is used.
- **Data anomalies retained in the main analysis** (and tested in phase R_SENS): solar-affected aggregates in Houses 11 and 21; House 11 fridge channel ON 0.22% of the time at about 1007 W; House 4 kettle ON 0.004%; House 21 dishwasher ON 13.1% at 356 W; combined kettle and toaster channels in Houses 17 and 21.
- **Reference house partition.** Stored explicitly because stratified group k-fold assignments can change between scikit-learn versions. All reference-partition experiments read it; phase B draws three further partitions with seeds 1, 2 and 3.
- **Leave-house-out hyperparameters** are fixed from the pooled tuning, which saw all houses; phase C re-tunes four models on the training houses only (mean change +0.003 in F1).

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `CLEAN_House*.csv not found` | Extract the archive (Section 3) and pass `--raw-dir`, or place `refit_raw/` next to the repository |
| `FileNotFoundError: data/features/...` | Run Step 2 first |
| Phase D or R_CNN logs "raw REFIT CSV files not found" | Put the raw files in one of the folders listed in Section 3 |
| LightGBM import error on macOS | `brew install libomp` |
| Memory errors for RF | Expected for unlimited depth on the full data; the final grid uses depth 12 |
| A long run stopped | Re-run the same command; finished work is skipped |
| `caffeinate: command not found` | Not on macOS; remove `caffeinate` from the shell scripts |

## 13. Limitations

A single dataset (REFIT); each leave-house-out fold tests only three or four houses; most leave-house-out results use hyperparameters fixed from pooled tuning (house-grouped tuning of four models changed scores by less than 0.01 on average, not repeated for RF, k-NN and SVM-RBF); the CNN is small and its architecture was not tuned, so it does not represent the best deep-learning performance under this protocol; the data anomalies listed in Section 11 were retained in the main analysis.

## 14. Citation and licence

If you use this code or these results, please cite the paper (reference to be added on publication), this repository (`CITATION.cff`) and the REFIT dataset:

D. Murray, L. Stankovic, V. Stankovic, An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study, Scientific Data 4 (2017) 160122. https://doi.org/10.1038/sdata.2016.122

Code: MIT licence (`LICENSE`). Result files: CC BY 4.0, consistent with the licence of the REFIT dataset.
