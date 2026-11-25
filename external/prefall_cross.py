import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#!/usr/bin/env python3
"""
Cross-dataset pre-impact evaluation: SisFall <-> KFall, same pipeline as prefall_v3c.py.

    python3 -I prefall_cross.py --kfall "<.../KFall Dataset/KFall Dataset>" --sis_cache real_v5/windows_cache.pkl --out cross_out

Design decisions (fixed before looking at any result):
  * KFall is 100 Hz; it is linearly resampled to 200 Hz so every window/filter constant of the SisFall pipeline is unchanged.
  * KFall has no second accelerometer -> the cross-dataset feature set drops f16-f18 (19 features: f1-f15, f19-f22).
  * Impact time for KFall = labelled Fall_impact_frame (not the 3 g rule); onset = labelled Fall_onset_frame.
  * Positive window = window ending 0-2.0 s BEFORE impact (as in SisFall).  NOTE: a window ending after fall onset already
    contains the start of the fall, so we also report detection BEFORE onset (true prediction) separately.
  * Thresholds always come from the TRAINING set (inner subject-wise OOF quantile at ADL-window FPR targets 2/5/10 %).
    "Oracle" rows set the threshold on the TEST ADL windows (shows ranking quality independent of calibration shift).
"""
import argparse, os, glob, re, json, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import prefall_v3c as P
warnings.filterwarnings("ignore")

FS = P.FS
COLS = P.L0 + P.L1 + P.L4          # 19 features, no second accelerometer

def kfall_items(root):
    lab = {}
    for f in sorted(glob.glob(os.path.join(root, "label_data", "*.xlsx"))):
        sid = re.match(r"(SA\d+)_label", os.path.basename(f)).group(1)
        l = pd.read_excel(f); l["Task Code (Task ID)"] = l["Task Code (Task ID)"].ffill()
        for _, r in l.iterrows():
            m = re.match(r"F(\d+) \((\d+)\)", str(r["Task Code (Task ID)"]))
            if m: lab[(sid, int(m.group(2)), int(r["Trial ID"]))] = (int(r["Fall_onset_frame"]), int(r["Fall_impact_frame"]))
    items = []
    for p in sorted(glob.glob(os.path.join(root, "sensor_data", "*", "*.csv"))):
        m = re.match(r"S(\d+)T(\d+)R(\d+)\.csv", os.path.basename(p))
        sid = "SA" + m.group(1); tid, rid = int(m.group(2)), int(m.group(3))
        isf = (sid, tid, rid) in lab
        if 20 <= tid <= 34 and not isf: continue                       # fall trial without a label: skip (counted below)
        items.append(dict(path=p, subject="K" + sid, act=f"T{tid:02d}", is_fall=int(isf),
                          onset_f=lab[(sid, tid, rid)][0] if isf else None, impact_f=lab[(sid, tid, rid)][1] if isf else None))
    return items

def _work(it):
    try:
        d = pd.read_csv(it["path"]); t = d["TimeStamp(s)"].values; t = t - t[0]
        tn = np.arange(0, t[-1], 1.0 / FS)
        acc = np.stack([np.interp(tn, t, d[c].values) for c in ("AccX", "AccY", "AccZ")], 1)
        gyr = np.stack([np.interp(tn, t, d[c].values) for c in ("GyrX", "GyrY", "GyrZ")], 1)
        F, T, _ = P.trial_windows(acc, gyr, acc.copy())
        imp = on = np.nan
        if it["is_fall"]: imp = (it["impact_f"] - 1) * 0.01; on = (it["onset_f"] - 1) * 0.01
        return it, F, T, imp, on, float(tn[-1])
    except Exception as ex:
        return it, None, None, np.nan, np.nan, 0.0

def build_kfall(root, cache):
    if os.path.exists(cache): return pd.read_pickle(cache)
    items = kfall_items(root); rows = []; bad = 0
    with Pool(2) as pool:
        for i, (it, F, T, imp, on, dur) in enumerate(pool.imap(_work, items, chunksize=8)):
            if F is None or len(F) == 0: bad += 1; continue
            if it["is_fall"]:
                if not np.isfinite(imp): bad += 1; continue
                keep = T <= imp
            else: keep = np.ones(len(T), bool)
            for f, t in zip(F[keep], T[keep]):
                rows.append([i, it["subject"], it["act"], it["is_fall"], t, imp, on, dur, *f])
            if i % 500 == 0: print(f"  kfall features {i}/{len(items)}", flush=True)
    df = pd.DataFrame(rows, columns=["trial", "subject", "act", "is_fall", "end", "impact", "onset", "dur"] + P.NAMES)
    df["ctx_prev_intensity"] = df.groupby("trial")["act_intensity"].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    df["ctx_prev_intensity"] = df["ctx_prev_intensity"].fillna(df["act_intensity"])
    df["dt"] = df["impact"] - df["end"]
    df["pos"] = ((df.is_fall == 1) & (df.dt <= P.HORIZON)).astype(int)
    df["train_neg"] = ((df.is_fall == 0) | ((df.is_fall == 1) & (df.dt >= P.GRAY))).astype(int)
    df["adl"] = (df.is_fall == 0).astype(int)
    df.attrs["bad"] = bad; df.to_pickle(cache); print("kfall table", df.shape, "unusable trials", bad); return df

# ------------------------------------------------------------------ evaluation ------------------
def alarms_per_hour(te, sc, thr, lock=5.0):
    n_al, hours = 0, 0.0
    d = te[te.adl == 1].assign(s=sc[(te.adl == 1).values])
    for _, g in d.groupby("trial"):
        g = g.sort_values("end"); hours += g.end.max() / 3600.0; last = -1e9
        for t_, s_ in zip(g.end.values, g.s.values):
            if s_ > thr and t_ - last >= lock: n_al += 1; last = t_
    return n_al / max(hours, 1e-9)

def pre_onset_det(te, sc, thr):
    """Share of fall trials with an alarm BEFORE fall onset (needs onset labels)."""
    if "onset" not in te or te["onset"].isna().all(): return np.nan
    f = te[(te.is_fall == 1)].assign(s=sc[(te.is_fall == 1).values])
    f = f[f.pos == 1]
    ok = f[(f.s > thr) & (f.end <= f.onset)].trial.nunique()
    return ok / max(f.trial.nunique(), 1)

def boot_auc(te_ev, s, B=200, seed=0):
    rng = np.random.default_rng(seed); subs = te_ev.subject.values; us = np.unique(subs)
    idx = {u: np.where(subs == u)[0] for u in us}; y = te_ev.pos.values; out = []
    for _ in range(B):
        pick = rng.choice(us, len(us)); ii = np.concatenate([idx[u] for u in pick])
        if y[ii].min() == y[ii].max(): continue
        out.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def evaluate(train, test, kind, cols, tag, rows):
    P.MODEL = kind; mdl = P.fit_rf(train, cols); sc_all = P.score_rf(mdl, test)
    ev = ((test.pos == 1) | (test.adl == 1)).values; te = test[ev]; s = sc_all[ev]
    auc = roc_auc_score(te.pos.values, s); lo, hi = boot_auc(te, s)
    adl_s = s[(te.adl == 1).values]
    for tg in P.FPR_TARGETS:
        for mode, thr in (("train-threshold", mdl["thr_fpr"][tg]), ("oracle-threshold", float(np.quantile(adl_s, 1 - tg)))):
            m = P.op_metrics(s, te.trial.values, te.pos.values == 1, te.adl.values == 1, te.dt.values, thr)
            rows.append(dict(setting=tag, model=kind, features=len(cols), target_fpr=tg, threshold=mode, AUC=round(auc, 3),
                             AUC_lo=round(lo, 3), AUC_hi=round(hi, 3), sens=round(m["sens"], 3), win_fpr=round(m["fpr_win"], 3),
                             adl_trial_alarm=round(m["fpr_trial"], 3), warn_s=round(m["warn"], 2),
                             alarms_per_h=round(alarms_per_hour(te, s, thr), 1),
                             pre_onset_det=round(pre_onset_det(te, s, thr), 3) if mode == "train-threshold" else None,
                             n_fall=m["n_fall"], n_adl_trials=m["n_adl"]))
    print(tag, kind, "AUC", round(auc, 3), (round(lo, 3), round(hi, 3)), flush=True)
    return mdl, te, s

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kfall", required=True); ap.add_argument("--sis_cache", required=True)
    ap.add_argument("--out", default="cross_out"); ap.add_argument("--smoke", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sis = pd.read_pickle(a.sis_cache); kf = build_kfall(a.kfall, os.path.join(a.out, "kfall_windows.pkl"))
    if a.smoke:
        keep_s = sorted(sis.subject.unique())[:a.smoke]; sis = sis[sis.subject.isin(keep_s)]
        keep_k = sorted(kf.subject.unique())[:a.smoke]; kf = kf[kf.subject.isin(keep_k)]
    print("SisFall", sis.shape, "KFall", kf.shape)
    print("KFall positives", kf.pos.sum(), "fall trials", kf[kf.is_fall == 1].trial.nunique(), "ADL trials", kf[kf.adl == 1].trial.nunique())
    rows = []
    for kind in ("rf", "hgb"):
        evaluate(sis, kf, kind, COLS, "train SisFall -> test KFall", rows)
        evaluate(kf, sis, kind, COLS, "train KFall -> test SisFall", rows)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "cross_results.csv"), index=False)
    # within-dataset references (subject-wise 10-fold CV), same 19 features
    ref = []
    for name, df in (("KFall within", kf), ("SisFall within", sis)):
        for kind in ("rf", "hgb"):
            P.MODEL = kind; res, _ = P.run_cv(df, "rf", COLS, 10, None)
            ref.append(dict(setting=name + " (10-fold subject-wise CV)", model=kind, AUC=round(res.auc.mean(), 3), AUC_sd=round(res.auc.std(), 3),
                            sens5=round(res["sens@5"].mean(), 3), win_fpr5=round(res["fpr_win@5"].mean(), 3)))
            print(ref[-1], flush=True)
            pd.DataFrame(ref).to_csv(os.path.join(a.out, "within_results.csv"), index=False)
    print("ALL DONE")

if __name__ == "__main__":
    main()
