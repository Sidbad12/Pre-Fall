# PreFall-Net

Code for "PreFall-Net: pre-impact fall warning from a waist-worn inertial sensor" (JIST-2340).

## Setup
    pip install numpy scipy pandas scikit-learn
    export SISFALL_ROOT=/path/to/SisFall_dataset      # SisFall is public; download it separately

## Layout
- `prefall_v3c.py` - main pipeline: causal filter, Madgwick global frame, 2 s windows, 22 features, baselines, subject-wise 10-fold CV. `--synthetic` runs a smoke test without data.
- `analysis/` - nested CV, paired/bootstrap tests, alarm-rate analysis, operating-point reconciliation, feature-group ablations (paper Sections 3.1 to 3.9).
- `external/` - KFall transfer, UMAFall placement, LTMM free-living false alarms, FallAllD evaluation, pooled leave-one-dataset-out training (Sections 3.10 to 3.12). These need KFall, UMAFall, LTMM and FallAllD downloaded separately; paths are set at the top of each script.
- `second_stage/` - impact-plus-stillness confirmation, stage-2-alone and activity-aware rules (Section 3.13). `two_stage.py` builds `mag_cache.pkl` on first run.

## Notes
- Several analysis scripts read intermediate files written by earlier ones (for example `analysis_causal/alarm/all_scores.pkl`); run `prefall_v3c.py` and `analysis/alarm_work.py` first.
- FallAllD axis mapping was chosen after a diagnostic (see manuscript Section 3.10).
- Datasets are not redistributed.

## Licence
MIT (see `LICENSE`).
