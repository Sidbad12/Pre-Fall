#!/usr/bin/env python3
"""
PreFall-Net v2 (rebuild). One script -> every number, table and figure in the revised paper.

    python prefall_v2.py --sisfall /path/to/SisFall_dataset --out results_v2

What changed vs the first rebuild (and why - put these in Methods):
  1. Madgwick filter is INITIALISED from the first 0.5 s of gravity. (SisFall's gravity axis is the sensor y axis;
     starting from "upright" left the global frame ~40 deg wrong after 5 s, corrupting every vertical/horizontal feature.)
  2. Evaluation is window-based and fair:
       positive window  = window ending within 2.0 s before impact (impact = first sample with |a| > 3 g)
       negative window  = any window of an ADL trial          (false-positive rate counted per ADL window)
       fall detected    = any positive window above threshold (sensitivity per fall trial)
       warning time     = impact time - end time of first positive window above threshold (max 2.0 s by definition)
     Earlier windows of fall trials (>= 3 s before impact) are extra NEGATIVES in training only.
  3. Subject-wise CV (GroupKFold): no subject in both train and test.
  4. Features f1-f4 as in the paper (global-frame definitions) + f5 jerk, f6 postural (tilt) angle, f7 energy ratio,
     f8 peak horizontal angular rate. Classifiers: (A) paper's weighted fusion of f1-f4, (B) Random Forest on f1-f8.
  5. Thresholds / weights are chosen on TRAINING subjects only (inner subject-wise CV for the Random Forest).
"""
import argparse, os, glob, json, re, warnings
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, lfilter, lfilter_zi
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
warnings.filterwarnings("ignore")
try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

FS = 200.0
WIN, STEP = int(2 * FS), int(0.5 * FS)              # 2 s window, 0.5 s hop (75 % overlap)
HORIZON, GRAY = 2.0, 3.0
RF_TREES = 100
MODEL = 'rf'
FPR_TARGETS = (0.02, 0.05, 0.10)
NAMES = ["step_irregularity", "trunk_sway", "foot_clearance", "direction_reversals",
         "jerk_rms", "tilt_max", "energy_ratio", "peak_sway_rate",
         # lever 1: pre-impact features (free-fall signature, short-window dynamics)
         "min_mag_1s", "freefall_frac_1s", "vvel_change_last05", "tilt_rate_max", "gyro_rms_last05", "jerk_last05", "tilt_delta_last05",
         # lever 3: second accelerometer (MMA8451Q)
         "mma_min_mag_1s", "mma_jerk_rms", "mma_energy_ratio",
         # lever 4: activity context
         "act_intensity", "gait_band_ratio", "ctx_prev_intensity", "dom_freq"]
NF = len(NAMES)
L0, L1, L3, L4 = list(range(8)), list(range(8, 15)), list(range(15, 18)), list(range(18, 22))
CORE = [0, 1, 2, 3]
ALPHA, BETA, GAMMA = 0.4, 0.5, 0.1                  # Eq. 7

# ------------------------------------------------------------------ orientation -------------------
@njit
def madgwick(acc, gyr_rad, fs, beta, q0, q1, q2, q3):
    n = acc.shape[0]; out = np.empty((n, 4)); dt = 1.0 / fs
    for i in range(n):
        ax, ay, az = acc[i, 0], acc[i, 1], acc[i, 2]
        gx, gy, gz = gyr_rad[i, 0], gyr_rad[i, 1], gyr_rad[i, 2]
        qd0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz); qd1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qd2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);  qd3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)
        nrm = np.sqrt(ax * ax + ay * ay + az * az)
        if nrm > 1e-9:
            ax /= nrm; ay /= nrm; az /= nrm
            f1 = 2 * (q1 * q3 - q0 * q2) - ax; f2 = 2 * (q0 * q1 + q2 * q3) - ay; f3 = 2 * (0.5 - q1 * q1 - q2 * q2) - az
            s0 = -2 * q2 * f1 + 2 * q1 * f2; s1 = 2 * q3 * f1 + 2 * q0 * f2 - 4 * q1 * f3
            s2 = -2 * q0 * f1 + 2 * q3 * f2 - 4 * q2 * f3; s3 = 2 * q1 * f1 + 2 * q2 * f2
            sn = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if sn > 1e-12:
                qd0 -= beta * s0 / sn; qd1 -= beta * s1 / sn; qd2 -= beta * s2 / sn; qd3 -= beta * s3 / sn
        q0 += qd0 * dt; q1 += qd1 * dt; q2 += qd2 * dt; q3 += qd3 * dt
        qn = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3); q0 /= qn; q1 /= qn; q2 /= qn; q3 /= qn
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = q0, q1, q2, q3
    return out

def gravity_init_quat(acc0):
    """Quaternion (body->global) that rotates the mean measured gravity vector onto +z."""
    a = acc0 / np.linalg.norm(acc0); z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, z); s = np.linalg.norm(axis); c = float(np.dot(a, z))
    if s < 1e-9: return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    th = np.arctan2(s, c); axis /= s
    return (float(np.cos(th / 2)), *(axis * np.sin(th / 2)))

def rot_matrices(q):
    w, x, y, z = q.T; R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

# ------------------------------------------------------------------ loading ----------------------
def load_sisfall_file(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip().rstrip(";")
            if ln: rows.append([float(v) for v in ln.split(",")[:9]])
    d = np.asarray(rows)
    return (d[:, 0:3] * (2 * 16 / 2 ** 13), d[:, 3:6] * (2 * 2000 / 2 ** 16),
            d[:, 6:9] * (2 * 8 / 2 ** 14))                                      # g (ADXL345), deg/s (ITG3200), g (MMA8451Q)

def list_sisfall(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)):
        m = re.match(r"([DF])(\d+)_(S[AE]\d+)_R(\d+)\.txt", os.path.basename(p))
        if m: out.append(dict(path=p, is_fall=int(m.group(1) == "F"), act=f"{m.group(1)}{m.group(2)}", subject=m.group(3)))
    return out

# ------------------------------------------------------------------ features ---------------------
_b, _a = butter(4, 5.0 / (FS / 2), btype="low")

def causal_lp(x):
    zi = lfilter_zi(_b, _a)[:, None] * x[0][None, :]
    return lfilter(_b, _a, x, axis=0, zi=zi)[0]

def trial_windows(acc, gyr, mma):
    mag_raw = np.linalg.norm(acc, axis=1); hit = np.where(mag_raw > 3.0)[0]
    impact = hit[0] / FS if len(hit) else np.nan
    accf, gyrf, mmaf = causal_lp(acc), causal_lp(gyr), causal_lp(mma)       # CAUSAL: no future samples (deployable)
    q0 = gravity_init_quat(accf[: int(0.5 * FS)].mean(0))
    q = madgwick(accf.astype(np.float64), np.deg2rad(gyrf).astype(np.float64), FS, 0.1, *q0)
    R = rot_matrices(q); ag = np.einsum("nij,nj->ni", R, accf); wg = np.einsum("nij,nj->ni", R, gyrf)
    a_vert = ag[:, 2] - 1.0; a_hor = ag[:, :2]; w_hor = np.linalg.norm(wg[:, :2], axis=1)
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1))); tilt_rate = np.abs(np.diff(tilt, prepend=tilt[0])) * FS
    mag = np.linalg.norm(accf, axis=1); dyn = mag - 1.0; jerk = np.linalg.norm(np.diff(accf, axis=0), axis=1) * FS
    mmag = np.linalg.norm(mmaf, axis=1); mdyn = mmag - 1.0; mjerk = np.linalg.norm(np.diff(mmaf, axis=0), axis=1) * FS
    h, q1s = int(0.5 * FS), int(1.0 * FS)
    F, T = [], []
    for s in range(0, len(acc) - WIN + 1, STEP):
        e = s + WIN
        pk, _ = find_peaks(mag[s:e], distance=int(0.3 * FS), prominence=0.05); ibi = np.diff(pk) / FS
        f1 = float(np.std(ibi) / np.mean(ibi)) if len(ibi) >= 3 and np.mean(ibi) > 0 else 0.0
        f2 = float(np.sqrt(np.mean(w_hor[s:e] ** 2)))
        f3 = float(a_vert[s:e].max() - a_vert[s:e].min())
        f4 = float(np.sum(np.linalg.norm(np.diff(a_hor[s:e], axis=0), axis=1) > 0.2))
        f5 = float(np.sqrt(np.mean(jerk[s:e - 1] ** 2))); f6 = float(tilt[s:e].max())
        f7 = float(np.sum(dyn[e - h:e] ** 2) / (np.sum(dyn[s:e] ** 2) + 1e-6)); f8 = float(w_hor[s:e].max())
        # lever 1
        g1 = float(mag[e - q1s:e].min()); g2 = float(np.mean(mag[e - q1s:e] < 0.6))
        g3 = float(np.sum(a_vert[e - h:e]) * 9.81 / FS); g4 = float(tilt_rate[s:e].max())
        g5 = float(np.sqrt(np.mean(w_hor[e - h:e] ** 2))); g6 = float(np.sqrt(np.mean(jerk[e - h - 1:e - 1] ** 2)))
        g7 = float(tilt[e - 1] - tilt[e - h])
        # lever 3
        m1 = float(mmag[e - q1s:e].min()); m2 = float(np.sqrt(np.mean(mjerk[s:e - 1] ** 2)))
        m3 = float(np.sum(mdyn[e - h:e] ** 2) / (np.sum(mdyn[s:e] ** 2) + 1e-6))
        # lever 4
        x = mag[s:e] - mag[s:e].mean(); P = np.abs(np.fft.rfft(x)) ** 2; fr = np.fft.rfftfreq(len(x), 1 / FS)
        tot = P[(fr >= 0.1) & (fr <= 20)].sum() + 1e-9; band = P[(fr >= 0.5) & (fr <= 3.5)].sum()
        sel = (fr >= 0.3) & (fr <= 5); c1 = float(np.std(mag[s:e])); c2 = float(band / tot); c4 = float(fr[sel][np.argmax(P[sel])])
        F.append([f1, f2, f3, f4, f5, f6, f7, f8, g1, g2, g3, g4, g5, g6, g7, m1, m2, m3, c1, c2, c1, c4]); T.append(e / FS)
    return np.asarray(F).reshape(-1, NF), np.asarray(T), impact

def build_table(items, cache):
    if os.path.exists(cache): return pd.read_pickle(cache)
    rows = []; dropped = 0
    for i, it in enumerate(items):
        try:
            acc, gyr, mma = (load_sisfall_file(it["path"]) if "path" in it else (it["acc"], it["gyr"], it["acc"]))
            F, T, imp = trial_windows(acc, gyr, mma)
        except Exception as ex:
            print("skip", it.get("path"), ex); continue
        if len(F) == 0: continue
        if it["is_fall"] and np.isnan(imp): dropped += 1; continue
        keep = T <= imp if it["is_fall"] else np.ones(len(T), bool)
        for f, t in zip(F[keep], T[keep]):
            rows.append([i, it["subject"], it["act"], it["is_fall"], t, imp, *f])
        if i % 300 == 0: print(f"  features {i}/{len(items)}", flush=True)
    df = pd.DataFrame(rows, columns=["trial", "subject", "act", "is_fall", "end", "impact"] + NAMES)
    # causal context: mean activity intensity over the previous 8 windows (4 s) of the same trial
    df["ctx_prev_intensity"] = df.groupby("trial")["act_intensity"].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    df["ctx_prev_intensity"] = df["ctx_prev_intensity"].fillna(df["act_intensity"])
    df.attrs["dropped_falls_no_impact"] = dropped
    df["dt"] = df["impact"] - df["end"]
    df["pos"] = ((df.is_fall == 1) & (df.dt <= HORIZON)).astype(int)
    df["train_neg"] = ((df.is_fall == 0) | ((df.is_fall == 1) & (df.dt >= GRAY))).astype(int)
    df["adl"] = (df.is_fall == 0).astype(int)
    df.to_pickle(cache); pd.Series({"dropped": dropped}).to_pickle(cache + ".meta"); return df

# ------------------------------------------------------------------ operating-point metrics -------
def op_metrics(score, tr_ids, is_pos, is_adl, dt, thr):
    """score/tr_ids/is_pos/is_adl/dt are per-window arrays of ONE evaluation set."""
    above = score > thr
    fp_win = above[is_adl].mean() if is_adl.any() else np.nan
    # fall-trial detection
    ids = tr_ids[is_pos]; ab = above[is_pos]; d = dt[is_pos]
    n_fall = len(np.unique(ids))
    if ab.any():
        g = pd.Series(d[ab]).groupby(ids[ab]).max()            # earliest crossing = largest time-to-impact
        n_det = len(g); wt_mean = float(g.mean())
    else:
        n_det = 0; wt_mean = 0.0
    sens = n_det / max(n_fall, 1)
    adl_ids = tr_ids[is_adl]; adl_ab = above[is_adl]
    fp_trial = len(np.unique(adl_ids[adl_ab])) / max(len(np.unique(adl_ids)), 1)
    return dict(sens=sens, fpr_win=fp_win, fpr_trial=fp_trial, warn=wt_mean,
                n_fall=n_fall, n_det=n_det, n_adl=len(np.unique(adl_ids)), n_adl_fp=len(np.unique(adl_ids[adl_ab])))

def objective(m):
    return ALPHA * m["fpr_win"] + BETA * (1 - m["sens"]) + GAMMA * max(0.0, 1.0 - m["warn"])           # Eq. 7

def pick_threshold(score, d, grid):
    best = (np.inf, None)
    for t in grid:
        m = op_metrics(score, d["trial"].values, d["pos"].values == 1, d["adl"].values == 1, d["dt"].values, t)
        if m["sens"] == 0: continue
        J = objective(m)
        if J < best[0]: best = (J, t)
    return best[1] if best[1] is not None else grid[-1]

# ------------------------------------------------------------------ classifiers -------------------
def weight_grid(k, step):
    n = int(round(1 / step)); out = []
    def rec(i, left, cur):
        if i == k - 1: out.append(cur + [left]); return
        for v in range(left + 1): rec(i + 1, left - v, cur + [v])
    rec(0, n, []); return np.asarray(out) * step

def fit_fusion(tr, cols, step):
    lo, hi = np.percentile(tr[[NAMES[c] for c in cols]].values, 1, 0), np.percentile(tr[[NAMES[c] for c in cols]].values, 99, 0)
    X = np.clip(100 * (tr[[NAMES[c] for c in cols]].values - lo) / np.maximum(hi - lo, 1e-9), 0, 100)
    sub = tr[(tr.pos == 1) | (tr.adl == 1)]; Xs = X[((tr.pos == 1) | (tr.adl == 1)).values]
    grid = np.arange(5, 100, 2.5); best = (np.inf, None, None)
    for w in weight_grid(len(cols), step):
        sc = Xs @ w
        for t in grid:
            m = op_metrics(sc, sub["trial"].values, sub["pos"].values == 1, sub["adl"].values == 1, sub["dt"].values, t)
            if m["sens"] == 0: continue
            J = objective(m)
            if J < best[0]: best = (J, w.copy(), t)
    return dict(w=best[1], thr=best[2], lo=lo, hi=hi, cols=cols)

def score_fusion(model, d):
    X = d[[NAMES[c] for c in model["cols"]]].values
    return np.clip(100 * (X - model["lo"]) / np.maximum(model["hi"] - model["lo"], 1e-9), 0, 100) @ model["w"]

def rf_fit(tr, cols, seed=0):
    t = tr[(tr.pos == 1) | (tr.train_neg == 1)]
    if MODEL == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        rf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                            l2_regularization=1.0, class_weight="balanced", random_state=seed)
    else:
        rf = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=20, max_depth=14, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=seed)
    rf.fit(t[[NAMES[c] for c in cols]].values, t["pos"].values); return rf

def fit_rf(tr, cols):
    sub = tr[(tr.pos == 1) | (tr.adl == 1)]
    # inner subject-wise CV for out-of-fold scores -> threshold
    oof = pd.Series(np.nan, index=tr.index); subs = tr.subject.values
    for a_, b_ in GroupKFold(3).split(tr, groups=subs):
        rf = rf_fit(tr.iloc[a_], cols); oof.iloc[b_] = rf.predict_proba(tr.iloc[b_][[NAMES[c] for c in cols]].values)[:, 1]
    m = ((tr.pos == 1) | (tr.adl == 1)).values
    thr = pick_threshold(oof.values[m], tr[m], np.arange(0.05, 0.96, 0.025))
    adl_oof = oof.values[(tr.adl == 1).values]
    thr_fpr = {t: float(np.quantile(adl_oof, 1 - t)) for t in FPR_TARGETS}
    return dict(rf=rf_fit(tr, cols), thr=thr, cols=cols, thr_fpr=thr_fpr)

def score_rf(model, d): return model["rf"].predict_proba(d[[NAMES[c] for c in model["cols"]]].values)[:, 1]

# ------------------------------------------------------------------ CV driver ---------------------
def run_cv(df, kind, cols, folds, wstep):
    rows, scored, ftype = [], [], []
    for k, (a_, b_) in enumerate(GroupKFold(folds).split(df, groups=df.subject.values)):
        tr, te = df.iloc[a_], df.iloc[b_]
        if te.pos.sum() == 0 or tr.pos.sum() < 20: continue
        if kind == "fusion":
            mdl = fit_fusion(tr, cols, wstep); sc = score_fusion(mdl, te); thr = mdl["thr"]; extra = dict(w=mdl["w"].round(2).tolist())
        else:
            mdl = fit_rf(tr, cols); sc = score_rf(mdl, te); thr = mdl["thr"]; extra = {}
        ev = ((te.pos == 1) | (te.adl == 1)).values; t2 = te[ev]; s2 = sc[ev]
        m = op_metrics(s2, t2.trial.values, t2.pos.values == 1, t2.adl.values == 1, t2.dt.values, thr)
        m["auc"] = roc_auc_score(t2.pos.values, s2); m["thr"] = thr; m["fold"] = k; m.update(extra)
        if kind == "rf":
            for tg, th in mdl["thr_fpr"].items():
                m2 = op_metrics(s2, t2.trial.values, t2.pos.values == 1, t2.adl.values == 1, t2.dt.values, th)
                for key in ("sens", "fpr_win", "fpr_trial", "warn"): m[f"{key}@{int(tg*100)}"] = m2[key]
        rows.append(m)
        out = t2[["trial", "act", "pos", "adl", "dt", "subject"]].copy(); out["score"] = s2; out["fold"] = k; out["thr"] = thr
        scored.append(out)
    return pd.DataFrame(rows), pd.concat(scored)

def kind_of(tag): return "rf" if "forest" in tag else "fusion"

def ms(x, pct=False):
    x = np.asarray(x, float); return (f"{100*np.nanmean(x):.1f} ± {100*np.nanstd(x):.1f} %" if pct else f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}")

def synth_items(n_subj=14, seed=0):
    rng = np.random.default_rng(seed); items = []; t = np.arange(0, 14, 1 / FS)
    for s in range(n_subj):
        for rep in range(4):
            for fall in (0, 1):
                hz = 1.8 + 0.1 * rng.standard_normal()
                acc = np.stack([0.15 * np.sin(2 * np.pi * hz * t), -1 + 0.35 * np.sin(2 * np.pi * hz * t), 0.1 * np.sin(np.pi * hz * t)], 1) \
                    + 0.03 * rng.standard_normal((len(t), 3))
                gyr = 15 * np.stack([np.sin(np.pi * hz * t), 0 * t, np.cos(np.pi * hz * t)], 1) + rng.standard_normal((len(t), 3))
                if fall:
                    i0 = int(rng.uniform(8, 11) * FS); ramp = np.clip((np.arange(len(t)) - (i0 - 1.5 * FS)) / (1.5 * FS), 0, 1)
                    gyr += 70 * ramp[:, None] * rng.standard_normal((len(t), 3)); acc += 0.4 * ramp[:, None] * rng.standard_normal((len(t), 3))
                    acc[i0:i0 + 5, 1] -= 4.0
                items.append(dict(acc=acc, gyr=gyr, is_fall=fall, act=("F01" if fall else "D01"), subject=f"SA{s:02d}"))
    return items

# ------------------------------------------------------------------ main (one pre-registered round) -----
CONFIGS = [  # fixed BEFORE looking at any result
    ("B0  RF  f1-f8 (baseline)",           "rf",  L0),
    ("C1  RF  + lever1 pre-impact feats",  "rf",  L0 + L1),
    ("C3  RF  + lever3 2nd accelerometer", "rf",  L0 + L3),
    ("C4  RF  + lever4 activity context",  "rf",  L0 + L4),
    ("C5  RF  all levers 1+3+4",           "rf",  L0 + L1 + L3 + L4),
    ("C2  HGB all levers 1+3+4 (lever 2)", "hgb", L0 + L1 + L3 + L4),
]

def main():
    global MODEL
    ap = argparse.ArgumentParser(); ap.add_argument("--sisfall"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results_v3"); ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    items = synth_items() if a.synthetic else list_sisfall(a.sisfall); assert items, "no trials found"
    df = build_table(items, os.path.join(a.out, "windows_cache.pkl"))
    ev = df[(df.pos == 1) | (df.adl == 1)]
    S = dict(falls=int(df[df.is_fall == 1].trial.nunique()), adl=int(df[df.is_fall == 0].trial.nunique()),
             single_feature_AUC={n: round(roc_auc_score(ev.pos, ev[n]), 3) for n in NAMES})
    print(S, flush=True)
    rows, scored = [], {}
    for name, model, cols in CONFIGS:
        MODEL = model; r, sc = run_cv(df, "rf", cols, a.folds, 0.1); scored[name] = sc
        row = {"config": name, "n_features": len(cols), "AUC": ms(r.auc)}
        for tg in FPR_TARGETS:
            k = int(round(tg * 100))
            row[f"sens@{k}%FPR"] = ms(r[f"sens@{k}"], True); row[f"achieved_win_FPR@{k}"] = ms(r[f"fpr_win@{k}"], True)
            row[f"ADL_trial_FA@{k}"] = ms(r[f"fpr_trial@{k}"], True); row[f"warn_s@{k}"] = ms(r[f"warn@{k}"])
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "config_comparison.csv"), index=False)
    S["configs"] = rows; json.dump(S, open(os.path.join(a.out, "summary_v3.json"), "w"), indent=2, default=float)
    # pooled ROC (window level) per config -> native Excel chart
    from openpyxl import Workbook
    from openpyxl.chart import ScatterChart, Reference, Series
    wb = Workbook(); ws = wb.active; ws.title = "ROC"; col = 1
    ch = ScatterChart(); ch.title = "ROC (window level, subject-wise CV)"; ch.x_axis.title = "False positive rate"; ch.y_axis.title = "True positive rate"
    ch.style = 13; ch.height, ch.width = 9, 15
    for name, sc in scored.items():
        f, t_, _ = roc_curve(sc.pos, sc.score); au = roc_auc_score(sc.pos, sc.score); step = max(1, len(f) // 300); f, t_ = f[::step], t_[::step]
        ws.cell(1, col, f"{name} FPR"); ws.cell(1, col + 1, f"{name} TPR")
        for i, (x, y) in enumerate(zip(f, t_), start=2): ws.cell(i, col, float(x)); ws.cell(i, col + 1, float(y))
        sr = Series(Reference(ws, min_col=col + 1, min_row=2, max_row=len(f) + 1), Reference(ws, min_col=col, min_row=2, max_row=len(f) + 1), title=f"{name} (pooled AUC={au:.3f})")
        sr.marker.symbol = "none"; ch.series.append(sr); col += 3
    ws.add_chart(ch, "A340"); ws2 = wb.create_sheet("Configs"); ws2.append(list(rows[0].keys()))
    for r_ in rows: ws2.append([str(v) for v in r_.values()])
    wb.save(os.path.join(a.out, "prefall_results_v3.xlsx"))
    print(pd.DataFrame(rows).T.to_string())

if __name__ == "__main__":
    main()
