import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#!/usr/bin/env python3
"""
UMAFall placement comparison (waist / wrist / chest / ankle) with the SAME pipeline, run at the SensorTags' native ~20 Hz.

    python3 prefall_uma.py --uma /path/to/UMAFall_Dataset --out uma_out

Fixed before looking at results:
  * SensorTag streams (acc g, gyro deg/s) are resampled to a uniform 20 Hz grid; pipeline constants are set for fs = 20 Hz.
  * Impact time = first sample > 3 g of the 200 Hz smartphone accelerometer (right pocket) - the only fast sensor. Fall trials
    without such a sample are dropped (counted).
  * Features f1-f15 and f19-f22 (19 features, no second accelerometer).  Positive window = ends 0-2.0 s before impact.
  * 19 subjects (13 fell) -> leave-one-subject-out; pooled out-of-fold scores; subject bootstrap for AUC.
  * Operating points use the pooled out-of-fold ADL-window quantile (an 'oracle-style' threshold): with 13 faller subjects
    per-fold thresholds are too noisy to be meaningful.
Caveat: UMAFall waist is 20 Hz, SisFall 200 Hz - results are not comparable to the SisFall numbers beyond ranking placements.
"""
import argparse, os, re, io, glob, json, warnings
import numpy as np, pandas as pd
from scipy.signal import butter
from sklearn.metrics import roc_auc_score
import prefall_v3c as P
warnings.filterwarnings("ignore")

FS = 20.0
P.FS = FS; P.WIN = int(2 * FS); P.STEP = int(0.5 * FS); P._b, P._a = butter(4, 5.0 / (FS / 2), btype="low")
LOAD_STATS = [0, 0]
PLACE = {1: "CHEST", 2: "WAIST", 3: "WRIST", 4: "ANKLE"}
BASE = list(P.NAMES)
COLS = P.L0 + P.L1 + P.L4

def load(path):
    rows = [l for l in open(path, errors="ignore") if re.match(r"^\d+;\d+;-?[\d.eE+-]+;-?[\d.eE+-]+;-?[\d.eE+-]+;\d;\d", l)]
    d = pd.read_csv(io.StringIO("".join(rows)), sep=";", header=None, usecols=range(7), names=["t", "n", "x", "y", "z", "type", "sid"], dtype=str)
    n0 = len(d)
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")      # some rows hold malformed numbers (e.g. '1.098.388.671.875'): dropped, not guessed
    d = d.dropna().reset_index(drop=True); LOAD_STATS[0] += n0; LOAD_STATS[1] += n0 - len(d)
    return d

def trial_rows(path):
    m = re.match(r"UMAFall_Subject_(\d+)_(ADL|Fall)_(.+)_(\d+)_20\d\d", os.path.basename(path))
    subj, kind, act = "U" + m.group(1), m.group(2), m.group(3); isf = int(kind == "Fall")
    d = load(path); d["t"] = d["t"] / 1000.0
    ph = d[(d.sid == 0) & (d.type == 0)]
    mag = np.linalg.norm(ph[["x", "y", "z"]].values, axis=1); hit = np.where(mag > 3.0)[0]
    impact = float(ph["t"].values[hit[0]]) if (isf and len(hit)) else np.nan
    if isf and not np.isfinite(impact): return None
    out = {}
    for sid, name in PLACE.items():
        a = d[(d.sid == sid) & (d.type == 0)]; g = d[(d.sid == sid) & (d.type == 1)]
        if len(a) < 40 or len(g) < 40: continue
        tmax = min(a.t.max(), g.t.max()); tn = np.arange(0, tmax, 1.0 / FS)
        acc = np.stack([np.interp(tn, a.t.values, a[c].values) for c in "xyz"], 1)
        gyr = np.stack([np.interp(tn, g.t.values, g[c].values) for c in "xyz"], 1)
        if len(tn) < P.WIN + 2: continue
        F, T, _ = P.trial_windows(acc, gyr, acc.copy())
        if len(F) == 0: continue
        keep = T <= impact if isf else np.ones(len(T), bool)
        out[name] = (F[keep], T[keep])
    return subj, act, isf, impact, out

def build(root, cache):
    if os.path.exists(cache): return pd.read_pickle(cache)
    tabs = {n: [] for n in PLACE.values()}; dropped = 0; n_fall = 0
    for i, p in enumerate(sorted(glob.glob(os.path.join(root, "*.csv")))):
        r = trial_rows(p)
        if r is None: dropped += 1; continue
        subj, act, isf, imp, out = r; n_fall += isf
        for name, (F, T) in out.items():
            for f, t in zip(F, T): tabs[name].append([i, subj, act, isf, t, imp, *f])
        if i % 100 == 0: print(f"  uma {i}", flush=True)
    dfs = {}
    for name, rows in tabs.items():
        df = pd.DataFrame(rows, columns=["trial", "subject", "act", "is_fall", "end", "impact"] + BASE)
        df["ctx_prev_intensity"] = df.groupby("trial")["act_intensity"].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
        df["ctx_prev_intensity"] = df["ctx_prev_intensity"].fillna(df["act_intensity"])
        df["dt"] = df["impact"] - df["end"]
        df["pos"] = ((df.is_fall == 1) & (df.dt <= P.HORIZON)).astype(int)
        df["train_neg"] = ((df.is_fall == 0) | ((df.is_fall == 1) & (df.dt >= P.GRAY))).astype(int)
        df["adl"] = (df.is_fall == 0).astype(int); dfs[name] = df
    pd.to_pickle((dfs, dropped), cache); print("fall trials without a >3 g phone sample (dropped):", dropped, "| malformed rows dropped:", LOAD_STATS[1], "of", LOAD_STATS[0]); return dfs, dropped

def alarms_per_hour(sc, thr, lock=5.0):
    n_al, hours = 0, 0.0
    for _, g in sc[sc.adl == 1].groupby("trial"):
        g = g.sort_values("end"); hours += g.end.max() / 3600.0; last = -1e9
        for t_, s_ in zip(g.end.values, g.score.values):
            if s_ > thr and t_ - last >= lock: n_al += 1; last = t_
    return n_al / max(hours, 1e-9)

def pooled_eval(sc, tag, kind, rows):
    y = sc.pos.values; s = sc.score.values; auc = roc_auc_score(y, s)
    rng = np.random.default_rng(0); subs = sc.subject.values; us = np.unique(subs); idx = {u: np.where(subs == u)[0] for u in us}; b = []
    for _ in range(300):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        if y[ii].min() != y[ii].max(): b.append(roc_auc_score(y[ii], s[ii]))
    lo, hi = np.percentile(b, [2.5, 97.5]); adl_s = s[sc.adl.values == 1]
    for tg in P.FPR_TARGETS:
        thr = float(np.quantile(adl_s, 1 - tg))
        m = P.op_metrics(s, sc.trial.values, y == 1, sc.adl.values == 1, sc.dt.values, thr)
        rows.append(dict(placement=tag, model=kind, AUC=round(auc, 3), AUC_lo=round(lo, 3), AUC_hi=round(hi, 3), target_fpr=tg,
                         sens=round(m["sens"], 3), win_fpr=round(m["fpr_win"], 3), adl_trial_alarm=round(m["fpr_trial"], 3),
                         warn_s=round(m["warn"], 2), alarms_per_h=round(alarms_per_hour(sc, thr), 1), n_fall=m["n_fall"], n_adl=m["n_adl"]))
    print(tag, kind, "AUC", round(auc, 3), (round(lo, 3), round(hi, 3)), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--uma", required=True); ap.add_argument("--out", default="uma_out"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dfs, dropped = build(a.uma, os.path.join(a.out, "uma_windows.pkl"))
    for n, d in dfs.items(): print(n, d.shape, "fall trials", d[d.pos == 1].trial.nunique(), "ADL trials", d[d.adl == 1].trial.nunique(), "subjects", d.subject.nunique())
    # combined waist+wrist table (aligned on trial and window end)
    w, r = dfs["WAIST"], dfs["WRIST"]
    r2 = r[["trial", "end"] + BASE].rename(columns={c: "wr_" + c for c in BASE})
    both = w.merge(r2, on=["trial", "end"], how="inner")
    extra = ["wr_" + c for c in BASE]; P.NAMES = BASE + extra; cols_both = COLS + [len(BASE) + c for c in COLS]
    rows = []
    for kind in ("rf", "hgb"):
        P.MODEL = kind
        for name in ("WAIST", "WRIST", "CHEST", "ANKLE"):
            _, sc = P.run_cv(dfs[name], "rf", COLS, dfs[name].subject.nunique(), None); sc["end"] = dfs[name].loc[sc.index, "end"]; pooled_eval(sc, name, kind, rows)
        _, sc = P.run_cv(both, "rf", cols_both, both.subject.nunique(), None); sc["end"] = both.loc[sc.index, "end"]; pooled_eval(sc, "WAIST+WRIST", kind, rows)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "uma_results.csv"), index=False)
    print("ALL DONE")

if __name__ == "__main__":
    main()
