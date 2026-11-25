# PreFall-Net: pre-impact fall warning from a waist-worn inertial sensor

Code behind the paper "PreFall-Net" (JIST-2340). It turns raw waist-worn accelerometer and gyroscope signals into a short-lead **pre-impact fall warning**, and tests how far that result holds on other datasets.

## Setup
    pip install numpy scipy pandas scikit-learn openpyxl
    export SISFALL_ROOT=/path/to/SisFall_dataset      # SisFall is public; download it separately
    mkdir -p real_v5 analysis_causal analysis_causal/alarm
Run all commands from the repository root. Datasets are not included.

## Step 1: the main pipeline (start here)
    python prefall_v3c.py --sisfall $SISFALL_ROOT --out real_v5
    python prefall_v3c.py --synthetic        # quick smoke test, no data needed

`prefall_v3c.py` does the core method: causal 4th-order Butterworth 5 Hz low-pass filter, Madgwick orientation filter in a gravity-referenced frame, 2 s windows every 0.5 s, 22 features, gradient boosting and the baselines, evaluated with subject-wise 10-fold cross-validation. It writes the headline tables, a `windows_cache.pkl` and a results workbook into `--out`. Most other scripts read `real_v5/windows_cache.pkl`.

## Step 2: what each other script does

### `analysis/` - SisFall deep-dives (paper Sections 3.1 to 3.9)
| Script | What it does | Main output |
|---|---|---|
| `prefall_analysis_c.py` | Shared analysis code: extra model types, baselines, held-out subjects, nested thresholds | baselines, alarms per hour, permutation importance, deployability |
| `nested_cv.py`, `nested_eval.py` | Model selection inside the training folds only, then compared with the fixed model | `item15_*` |
| `boot_paired.py`, `boot_all2000.py`, `boot_nn.py` | Paired subject-level bootstrap tests and confidence intervals | `item9*`, `item18*` |
| `extra_checks.py` | Warning-time percentiles and the sensitivity versus alarms-per-hour curve | `item11`, `item12` |
| `reconcile.py` | Pooled-quantile operating-point curve and reconciliation with the per-fold threshold | `item16*` |
| `alarm_work.py`, `alarm_eval.py` | False alarms per ADL hour, young versus elderly, alarm variants | `item19*` |
| `f21_ablation.py`, `run_ablate_c.py` | Removing feature groups to see what each contributes | `item13`, `ablate_causal.csv` |

### `external/` - other datasets (Sections 3.10 to 3.12; each needs its dataset downloaded)
| Script | What it does |
|---|---|
| `prefall_cross.py` | SisFall to KFall and KFall to SisFall transfer |
| `prefall_uma.py`, `uma_common.py`, `uma_wrist_paired.py` | UMAFall: waist, wrist, chest, ankle placement and the paired waist-plus-wrist test |
| `train_ltmm_model.py` | Trains the 19-feature model used to count false alarms on LTMM older adults (no falls in that dataset) |
| `fallalld_eval.py` | FallAllD evaluation (waist logger), with the axis-mapping diagnostic. Ran on Kaggle; paths are at the top |
| `lodo.py` | Pooled leave-one-dataset-out training across SisFall, KFall and FallAllD. Set `FALLALLD_FEATS` to the saved FallAllD feature file |

### `second_stage/` - the impact-confirmation stage (Section 3.13)
| Script | What it does | Main output |
|---|---|---|
| `two_stage.py` | Impact-plus-stillness confirmation of a stage-1 alarm. Builds `mag_cache.pkl` (raw acceleration magnitude per trial) on first run | `item22_two_stage_folds.csv` |
| `stage2_alone.py` | The same check used on its own, without stage 1. Run after `two_stage.py` because it reads `mag_cache.pkl` | `item22_stage2_alone.csv` |
| `export_raw.py` | Exports 50 Hz raw-signal windows for the deep-learning baselines (CNN/GRU), which were trained on Kaggle | `kaggle_pack/` |
| `activity_gate.py`, `activity_thr.py` | Activity-aware rules (motion gate, per-state thresholds); a negative result | `item21_*` |

## Reading the result files
Outputs named `itemN_*` are the numbered result tables behind the paper. For example `item4` gives the activities with the most false alarms and `item5` the sensitivity by fall type.

## Licence
MIT (see `LICENSE`).
