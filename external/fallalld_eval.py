import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#!/usr/bin/env python3
"""
External test of PreFall-Net on FallAllD (waist device, 238 Hz raw; Kaggle: 'Derived FallAllD Dataset' -> FallAllD.pkl).
Uses the 19-feature SisFall-trained model (ltmm_model.pkl) with SisFall thresholds, unchanged.  Needs prefall_v3c.py + ltmm_model.pkl alongside.
  python3 fallalld_eval.py --pkl /kaggle/input/.../FallAllD.pkl --device Waist --out fa_out
Prints diagnostics first (devices, activity ids, units check). FALLS = ActivityID >= 100 (FallAllD convention; override with --fall_min).
"""
import argparse, os, sys, json, pickle, itertools, warnings, glob
CODE = "/kaggle/input/datasets/sid155/dataset"      # folder with fallalld_eval.py, prefall_v3c.py, ltmm_model.pkl
PKL = "/kaggle/input/datasets/sankalpsinghvishen/derived-fallalld-dataset/FallAllD.pkl"
for _p in (CODE, os.path.dirname(os.path.abspath(__file__))): sys.path.insert(0, _p)
import numpy as np, pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import prefall_v3c as P
FS_RAW = 238.0
def _lp(x, fc, fs, order=2):
    b, a = butter(order, fc / (fs / 2)); zi = lfilter_zi(b, a)[:, None] * x[0][None, :]; return lfilter(b, a, x, axis=0, zi=zi)[0]
PERMS = [(p, s) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)]
def calibrate_axes(segs, fs):
    U, DU, W = [], [], []
    for acc, gyr in segs:
        u = _lp(acc, 2.0, fs); u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
        U.append(u[20:-20]); DU.append((np.gradient(u, axis=0) * fs)[20:-20]); W.append(np.deg2rad(_lp(gyr, 2.0, fs))[20:-20])
    u = np.concatenate(U); du = np.concatenate(DU); w0 = np.concatenate(W)
    nrm = np.linalg.norm(w0, axis=1); mov = nrm > np.quantile(nrm, 0.85); res = []
    for p, s_ in PERMS:
        w = w0[:, list(p)] * np.array(s_); pred = -np.cross(w, u)
        num = np.sum(du[mov] * pred[mov]); den = np.sqrt(np.sum(du[mov] ** 2) * np.sum(pred[mov] ** 2)) + 1e-12; res.append((num / den, p, s_))
    res.sort(key=lambda t: -t[0]); ident = [r for r in res if r[1] == (0, 1, 2) and r[2] == (1, 1, 1)][0][0]
    return res[0][1], res[0][2], res[0][0], res[1][0], ident
def resample(x, f_from, f_to):
    n = len(x); t0 = np.arange(n) / f_from; t1 = np.arange(0, t0[-1], 1.0 / f_to)
    return np.stack([np.interp(t1, t0, x[:, k]) for k in range(x.shape[1])], 1)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkl", default=PKL); ap.add_argument("--device", default="Waist"); ap.add_argument("--out", default="/kaggle/working/fa_out")
    ap.add_argument("--fall_min", type=int, default=100); ap.add_argument("--acc_scale", type=float, default=8.0 / 32768); ap.add_argument("--gyr_scale", type=float, default=0.070)
    ap.add_argument("--force_identity", action="store_true"); ap.add_argument("--perm"); ap.add_argument("--sign"); ap.add_argument("--scan", action="store_true"); ap.add_argument("--model", default=os.path.join(CODE, "ltmm_model.pkl")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pk = a.pkl if os.path.exists(a.pkl) else glob.glob("/kaggle/input/**/FallAllD.pkl", recursive=True)[0]; print("using", pk)
    D = pd.read_pickle(pk); print("rows", len(D), "devices", D.Device.value_counts().to_dict())
    D = D[D.Device.str.lower() == a.device.lower()].reset_index(drop=True)
    D["fall"] = (D.ActivityID >= a.fall_min).astype(int)
    print("device", a.device, "trials", len(D), "subjects", D.SubjectID.nunique(), "falls", int(D.fall.sum()), "ADL", int((1 - D.fall).sum()))
    print("activity ids (n trials):", D.groupby("ActivityID").size().to_dict())
    print("trial lengths (samples):", D.Acc.apply(len).describe().round(0).to_dict())
    for c in ("activity_info.pkl",):
        f = os.path.join(os.path.dirname(pk), c)
        if os.path.exists(f):
            try:
                ai = pd.read_pickle(f); print("activity_info:\n", (ai.to_string() if hasattr(ai, "to_string") else "\n".join(f"{k}: {v}" for k, v in ai.items())))
            except Exception as e: print("activity_info unreadable", e)
    model = pickle.load(open(a.model, "rb")); thr = model["thr_fpr"]
    acc = [t.astype(float) * a.acc_scale for t in D.Acc]; gyr = [t.astype(float) * a.gyr_scale for t in D.Gyr]
    # units check: median |a| in ADL trials should be ~1 g
    med = np.median([np.median(np.linalg.norm(x, axis=1)) for x, f in zip(acc, D.fall) if f == 0]); print("median |acc| in ADL trials (g):", round(float(med), 3), "(should be ~1.0)")
    if abs(med - 1) > 0.15: print("WARNING: accelerometer scale looks wrong; use --acc_scale"); 
    # axis check pooled over up to 150 ADL trials
    rg = np.random.default_rng(0); idx = rg.choice(np.where(D.fall.values == 0)[0], size=min(150, int((1 - D.fall).sum())), replace=False)
    segs = [(resample(acc[i], FS_RAW, 100.0), resample(gyr[i], FS_RAW, 100.0)) for i in idx]
    perm, sgn, s0, s1, sid = calibrate_axes(segs, 100.0); print("axis check: perm", perm, "sign", sgn, "score", round(s0, 3), "runner-up", round(s1, 3), "identity", round(sid, 3))
    if a.force_identity: perm, sgn = (0, 1, 2), (1, 1, 1)
    if a.perm: perm = tuple(int(x) for x in a.perm.split(",")); sgn = tuple(int(x) for x in a.sign.split(","))
    if a.scan:      # diagnostic: window AUC under the best physical mappings (labels NOT used to choose the mapping for the main run)
        sub = list(np.where(D.fall.values == 1)[0]) + list(rg.choice(np.where(D.fall.values == 0)[0], min(500, int((1 - D.fall).sum())), replace=False)); mo = pickle.load(open(a.model, "rb"))
        cands = [(perm, sgn), ((0, 1, 2), (1, 1, 1)), ((0, 1, 2), (-1, -1, -1))]
        for pm, sg in cands:
            Y, S_ = [], []
            for i in sub:
                a2 = resample(acc[i], FS_RAW, P.FS); g2 = resample(gyr[i][:, list(pm)] * np.array(sg), FS_RAW, P.FS)
                F, T, imp = P.trial_windows(a2, g2, a2.copy())
                if len(F) == 0: continue
                F = pd.DataFrame(F, columns=P.NAMES); it = F["act_intensity"].values
                F["ctx_prev_intensity"] = pd.Series(it).shift(1).rolling(8, min_periods=1).mean().fillna(pd.Series(it)).values
                sc = mo["model"].predict_proba(F[mo["names"]].values)[:, 1]
                if D.fall.values[i] == 1:
                    if not np.isfinite(imp): continue
                    dt = imp - T; m = (dt >= 0) & (dt <= 2.0); Y += [1] * int(m.sum()); S_ += list(sc[m])
                else: Y += [0] * len(sc); S_ += list(sc)
            print("SCAN perm", pm, "sign", sg, "window AUC", round(roc_auc_score(Y, S_), 3), flush=True)
        return
    rows = []; WIN = []; FEAT = []
    for i, r in D.iterrows():
        a2 = resample(acc[i], FS_RAW, P.FS); g2 = resample(gyr[i][:, list(perm)] * np.array(sgn), FS_RAW, P.FS)
        F, T, impact = P.trial_windows(a2, g2, a2.copy())
        if len(F) == 0: continue
        F = pd.DataFrame(F, columns=P.NAMES); inten = F["act_intensity"].values
        F["ctx_prev_intensity"] = pd.Series(inten).shift(1).rolling(8, min_periods=1).mean().fillna(pd.Series(inten)).values
        sc = model["model"].predict_proba(F[model["names"]].values)[:, 1]; mag = np.linalg.norm(a2, axis=1)
        if r.fall == 1:
            if not np.isfinite(impact): continue      # no sample above 3 g: excluded, as in SisFall
            dt = impact - T; m = (dt >= 0) & (dt <= 2.0); t_use, s_use = T[m], sc[m]; dts = dt[m]; FEAT.append(F.loc[m, model["names"]].values.astype(np.float32))
        else: t_use, s_use, dts = T, sc, np.full(len(T), np.nan); FEAT.append(F[model["names"]].values.astype(np.float32))
        for t_, s_, d_ in zip(t_use, s_use, dts): WIN.append((i, int(r.SubjectID), int(r.ActivityID), int(r.fall), t_, s_, d_))
        rows.append(dict(i=i, subject=int(r.SubjectID), act=int(r.ActivityID), fall=int(r.fall), impact=impact, dur=len(a2) / P.FS))
        MAGS[i] = mag.astype(np.float32)
    W = pd.DataFrame(WIN, columns=["trial", "subject", "act", "fall", "end", "score", "dt"]); meta = pd.DataFrame(rows).set_index("i")
    W.to_pickle(os.path.join(a.out, "fallalld_windows.pkl")); FA = pd.DataFrame(np.concatenate(FEAT), columns=model["names"]); FA = pd.concat([W.reset_index(drop=True), FA], axis=1); FA.to_pickle(os.path.join(a.out, "fallalld_feats.pkl")); print("saved features", FA.shape); print("windows", len(W), "fall trials used", int(meta.fall.sum()), "ADL trials", int((1 - meta.fall).sum()))
    pos = W[W.fall == 1]; adl = W[W.fall == 0]; y = W.fall.values
    out = dict(n_subjects=int(meta.subject.nunique()), n_fall_trials=int(meta.fall.sum()), n_adl_trials=int((1 - meta.fall).sum()))
    out["auc_window"] = float(roc_auc_score(y, W.score.values))
    subs = W.subject.unique(); bs = []
    for _ in range(1000):
        c = rg.choice(subs, len(subs)); Wb = pd.concat([W[W.subject == s] for s in c])
        if Wb.fall.nunique() == 2: bs.append(roc_auc_score(Wb.fall, Wb.score))
    out["auc_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    hours = meta[meta.fall == 0].dur.sum() / 3600
    def alarms(df, th, lock=5.0):
        n = 0; per = {}
        for tr, g in df.groupby("trial"):
            last = -1e9; c = 0
            for t_, s_ in zip(g.end.values, g.score.values):
                if s_ > th and t_ - last >= lock: c += 1; last = t_
            per[tr] = c
        return per
    def stage2(trial, t_al, G=3.5, s0=0.2, tw=2.5):
        m = MAGS[trial]; i0 = int(round(t_al * P.FS)); i1 = min(len(m), i0 + int(tw * P.FS))
        if i1 - i0 < 5: return False
        seg = m[i0:i1]; j = int(np.argmax(seg)); tp = i0 + j; a_ = tp + int(0.6 * P.FS); b_ = min(len(m), tp + int(2.6 * P.FS))
        sd = np.std(m[a_:b_]) if b_ - a_ >= int(1.0 * P.FS) else 0.0
        return float(seg[j]) >= G and sd < s0
    res = {}
    # thresholds: (1) SisFall-derived, unchanged; (2) recalibrated leave-one-subject-out to the same ADL-window FPR
    for tg, th in list(thr.items()) + [("recal5", None)]:
        if th is None:
            ths = {s: float(np.quantile(adl[adl.subject != s].score, 0.95)) for s in subs}
            det = []; wf = []; pa = {}; wd = []
            for s in subs:
                Wf = pos[pos.subject == s]; Wa = adl[adl.subject == s]
                for tr, g in Wf.groupby("trial"): det.append(float((g.score > ths[s]).any()))
                wf += list((Wa.score > ths[s]).astype(float))
            res["recal5_LOSO"] = dict(window_fpr=float(np.mean(wf)), fall_sens=float(np.mean(det)))
            continue
        al_pos = alarms(pos, th); al_adl = alarms(adl, th)
        sens = np.mean([al_pos.get(tr, 0) > 0 for tr in meta[meta.fall == 1].index])
        warn = []
        for tr, g in pos.groupby("trial"):
            gg = g[g.score > th]
            if len(gg): warn.append(float(gg.dt.max()))
        # two-stage at this threshold
        conf_f = 0; conf_a = 0
        for tr, g in pos.groupby("trial"):
            last = -1e9; ok = False
            for t_, s_ in zip(g.end.values, g.score.values):
                if s_ > th and t_ - last >= 5.0:
                    last = t_
                    if stage2(tr, t_): ok = True
            conf_f += ok
        for tr, g in adl.groupby("trial"):
            last = -1e9
            for t_, s_ in zip(g.end.values, g.score.values):
                if s_ > th and t_ - last >= 5.0:
                    last = t_
                    if stage2(tr, t_): conf_a += 1
        res[f"thr_fpr{int(round(tg*100))}"] = dict(threshold=float(th), window_fpr=float((adl.score > th).mean()), fall_sens=float(sens), median_warning_s=float(np.median(warn)) if warn else None,
            alarms_per_adl_hour=float(sum(al_adl.values()) / hours), adl_trials_alarmed=float(np.mean([v > 0 for v in al_adl.values()])),
            two_stage_fall_sens=float(conf_f / meta.fall.sum()), two_stage_alerts_per_adl_hour=float(conf_a / hours))
    # stage 2 alone (impact + stillness anywhere in the clip), same fixed limits
    def s2_alone(tr):
        m = MAGS[tr]; idx = np.where(m >= 3.5)[0]; last = -1e9; c = 0
        for i in idx:
            if i - last >= 5 * P.FS:
                last = i; j = i + int(1.0 * P.FS); seg = m[i:j]; tp = i + int(np.argmax(seg)); a_ = tp + int(0.6 * P.FS); b_ = min(len(m), tp + int(2.6 * P.FS))
                if (np.std(m[a_:b_]) if b_ - a_ >= int(1.0 * P.FS) else 0.0) < 0.2: c += 1
        return c
    res["stage2_alone"] = dict(fall_sens=float(np.mean([s2_alone(tr) > 0 for tr in meta[meta.fall == 1].index])), alerts_per_adl_hour=float(sum(s2_alone(tr) for tr in meta[meta.fall == 0].index) / hours))
    out["results"] = res; out["adl_hours"] = float(hours); out["axis_check"] = dict(perm=perm, sign=sgn, score=float(s0), identity=float(sid))
    json.dump(out, open(os.path.join(a.out, "fallalld_results.json"), "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
    # per-activity false alarms and per-fall-type sensitivity at the 5% threshold
    th = thr[0.05]; al_adl = alarms(adl, th); al_pos = alarms(pos, th)
    pa = pd.DataFrame(dict(act=[meta.loc[t, "act"] for t in al_adl], alarmed=[v > 0 for v in al_adl.values()])).groupby("act").alarmed.agg(["mean", "size"]).sort_values("mean", ascending=False)
    pf = pd.DataFrame(dict(act=[meta.loc[t, "act"] for t in meta[meta.fall == 1].index], det=[al_pos.get(t, 0) > 0 for t in meta[meta.fall == 1].index])).groupby("act").det.agg(["mean", "size"])
    pa.to_csv(os.path.join(a.out, "adl_alarm_share_by_activity.csv")); pf.to_csv(os.path.join(a.out, "fall_sensitivity_by_activity.csv")); print(pa.head(12)); print(pf)
    print("ALL DONE")
MAGS = {}
if __name__ == "__main__": main()
