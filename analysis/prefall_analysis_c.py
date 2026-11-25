import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#!/usr/bin/env python3
"""
PreFall-Net validation round (items 1-8). Reads the window cache produced by prefall_v3.py.
    python prefall_analysis.py --cache real_v4/windows_cache.pkl --sisfall /path/SisFall_dataset --out analysis_out
Stages checkpoint to disk, so a crash never loses finished work.
  1 same-dataset baselines   2 locked hold-out + subject-bootstrap CIs   3 alarms/hour with lockout
  4 false alarms by ADL       5 sensitivity by fall type                  6 permutation importance
  7 deployability timing      8 young vs elderly
"""
import sys, os, json, time, pickle, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import prefall_v3c as P
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

ACT = {"D01": "Walking slowly", "D02": "Walking quickly", "D03": "Jogging slowly", "D04": "Jogging quickly",
       "D05": "Stairs slowly", "D06": "Stairs quickly", "D07": "Sit half-height chair slowly", "D08": "Sit half-height chair quickly",
       "D09": "Sit low chair slowly", "D10": "Sit low chair quickly", "D11": "Sit, try to get up, collapse into chair",
       "D12": "Sit, lie slowly, sit again", "D13": "Sit, lie quickly, sit again", "D14": "Back to lateral position and back",
       "D15": "Bend at knees and get up", "D16": "Bend without bending knees and get up", "D17": "Get into car, sit, get out",
       "D18": "Stumble while walking", "D19": "Gentle jump",
       "F01": "Forward fall walking (slip)", "F02": "Backward fall walking (slip)", "F03": "Lateral fall walking (slip)",
       "F04": "Forward fall walking (trip)", "F05": "Forward fall jogging (trip)", "F06": "Vertical fall walking (fainting)",
       "F07": "Fall walking, hands on table (fainting)", "F08": "Forward fall trying to get up", "F09": "Lateral fall trying to get up",
       "F10": "Forward fall trying to sit down", "F11": "Backward fall trying to sit down", "F12": "Lateral fall trying to sit down",
       "F13": "Forward fall sitting (fainting/asleep)", "F14": "Backward fall sitting (fainting/asleep)", "F15": "Lateral fall sitting (fainting/asleep)"}
ALL = list(range(P.NF)); TGT = (2, 5, 10)

# ------------------------------------------------------------------ extra model kinds (same inner-CV threshold machinery) ----
class Wrap:
    def __init__(self, f): self.f = f
    def predict_proba(self, X): s = self.f(X); return np.c_[1 - s, s]

def sigmoid(z): return 1 / (1 + np.exp(-z))
_orig_rf_fit = P.rf_fit

def rf_fit_ext(tr, cols, seed=0):
    k = P.MODEL
    if k in ("rf", "hgb"): return _orig_rf_fit(tr, cols, seed)
    t = tr[(tr.pos == 1) | (tr.train_neg == 1)]
    X, y = t[[P.NAMES[c] for c in cols]].values, t["pos"].values
    if k == "rule_ff":      return Wrap(lambda Z: np.clip(1 - Z[:, 0], 0, 1))                  # low |a| in last 1 s -> fall
    if k == "rule_tilt":    return Wrap(lambda Z: np.clip(Z[:, 0] / 180.0, 0, 1))              # large tilt -> fall
    rng = np.random.default_rng(seed)
    if k == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")); m.fit(X, y); return m
    if k in ("svm", "mlp"):
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]; n_tot = 20000 if k == "svm" else 60000
        idx = np.r_[pos, rng.choice(neg, size=min(len(neg), max(n_tot - len(pos), 1000)), replace=False)]
        if k == "svm":
            m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")); m.fit(X[idx], y[idx])
            return Wrap(lambda Z: sigmoid(m.decision_function(Z)))
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=60, early_stopping=True, random_state=seed))
        m.fit(X[idx], y[idx]); return m
    raise ValueError(k)
P.rf_fit = rf_fit_ext

# ------------------------------------------------------------------ helpers ----------------------
def collect(df, kind, cols, folds=10):
    """Subject-wise CV; returns pooled evaluation windows with out-of-fold score and the fold's fixed-FPR thresholds."""
    P.MODEL = kind; out = []
    for k, (a, b) in enumerate(GroupKFold(folds).split(df, groups=df.subject.values)):
        tr, te = df.iloc[a], df.iloc[b]
        if te.pos.sum() == 0: continue
        mdl = P.fit_rf(tr, cols); d = te[["trial", "subject", "act", "is_fall", "pos", "adl", "dt", "end"]].copy()
        d["score"] = P.score_rf(mdl, te); d["fold"] = k
        for tg, th in mdl["thr_fpr"].items(): d[f"thr{int(round(tg * 100))}"] = th
        out.append(d[(d.pos == 1) | (d.adl == 1)])
        print(f"   [{kind}] fold {k}", flush=True)
    return pd.concat(out)

def fold_metrics(sc, k):
    rows = []
    for f, g in sc.groupby("fold"):
        m = P.op_metrics(g.score.values, g.trial.values, g.pos.values == 1, g.adl.values == 1, g.dt.values, g[f"thr{k}"].iloc[0])
        m["auc"] = roc_auc_score(g.pos, g.score); rows.append(m)
    return pd.DataFrame(rows)

def summarise(sc):
    row = {"AUC": P.ms(fold_metrics(sc, 5).auc)}
    for k in TGT:
        r = fold_metrics(sc, k)
        row[f"sens@{k}%"] = P.ms(r.sens, True); row[f"winFPR@{k}"] = P.ms(r.fpr_win, True)
        row[f"ADLtrialFA@{k}"] = P.ms(r.fpr_trial, True); row[f"warn@{k}"] = P.ms(r.warn)
    return row

def alarms_per_hour(sc, k, lock=5.0):
    n_al, hours = 0, 0.0
    for tid, g in sc[sc.adl == 1].groupby("trial"):
        g = g.sort_values("end"); hours += g.end.max() / 3600.0; last = -1e9
        for t_, s_, th in zip(g.end.values, g.score.values, g[f"thr{k}"].values):
            if s_ > th and t_ - last >= lock: n_al += 1; last = t_
    return n_al / max(hours, 1e-9), n_al, hours

def per_subject_counts(sc, k):
    """Per-subject sufficient statistics for the subject bootstrap at fixed-FPR target k."""
    rec = {}
    for s_, g in sc.groupby("subject"):
        g = g.assign(ab=g.score > g[f"thr{k}"])
        pos = g[g.pos == 1]; adl = g[g.adl == 1]
        det = pos[pos.ab].groupby("trial").dt.max()
        rec[s_] = dict(nf=pos.trial.nunique(), nd=len(det), wsum=float(det.sum()), nw=len(adl), nwa=int(adl.ab.sum()),
                       nt=adl.trial.nunique(), nta=adl[adl.ab].trial.nunique())
    return pd.DataFrame(rec).T

def boot(sc, k, B=500, seed=0):
    rng = np.random.default_rng(seed); C = per_subject_counts(sc, k); subs = C.index.values
    out = []
    for _ in range(B):
        S = C.loc[rng.choice(subs, len(subs))].sum()
        out.append([S.nd / max(S.nf, 1), S.nwa / max(S.nw, 1), S.nta / max(S.nt, 1), S.wsum / max(S.nd, 1)])
    o = np.array(out); ci = lambda x: f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"
    return dict(sens_CI=ci(o[:, 0]), winFPR_CI=ci(o[:, 1]), ADLtrialFA_CI=ci(o[:, 2]), warn_CI=ci(o[:, 3]))

def auc_ci(sc, B=300, seed=0):
    rng = np.random.default_rng(seed); idx = {s_: np.where(sc.subject.values == s_)[0] for s_ in sc.subject.unique()}
    subs = np.array(list(idx)); y, s = sc.pos.values, sc.score.values; a = []
    for _ in range(B):
        ii = np.concatenate([idx[x] for x in rng.choice(subs, len(subs))]); a.append(roc_auc_score(y[ii], s[ii]))
    return f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"

def ck(path, fn):
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    r = fn(); pickle.dump(r, open(path, "wb")); return r

# ------------------------------------------------------------------ main --------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--sisfall")
    ap.add_argument("--out", default="analysis_out"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_pickle(a.cache); R = {}
    CFG = {"C1_RF_lever1": ("rf", P.L0 + P.L1), "C2_HGB_all": ("hgb", ALL)}
    BASE = {"Rule: free-fall (min |a| last 1 s)": ("rule_ff", [8]), "Rule: max tilt": ("rule_tilt", [5]),
            "Logistic regression (22 feat)": ("lr", ALL), "SVM-RBF (22 feat)": ("svm", ALL), "MLP (22 feat)": ("mlp", ALL),
            "RF baseline (8 feat)": ("rf", P.L0)}

    # ---- stage A: out-of-fold scores for the two headline models (reused by items 2,3,4,5,8)
    sc = {}
    for name, (kind, cols) in CFG.items():
        print("== collect", name, flush=True); sc[name] = ck(os.path.join(a.out, f"oof_{name}.pkl"), lambda: collect(df, kind, cols))
    # ---- item 1: same-dataset baselines (same folds, same threshold machinery)
    rows = []
    for name, (kind, cols) in BASE.items():
        print("== baseline", name, flush=True)
        s_ = ck(os.path.join(a.out, f"oof_base_{kind}_{len(cols)}.pkl"), lambda: collect(df, kind, cols)); r = summarise(s_); r["model"] = name; rows.append(r)
    for name, s_ in sc.items(): r = summarise(s_); r["model"] = name; rows.append(r)
    print("== paper's original fusion (f1-f4, weights grid)", flush=True)
    P.FPR_TARGETS = (0.02, 0.05, 0.10)
    rf_, _ = P.run_cv(df, "fusion", P.CORE, 10, 0.1)
    rows.append({"model": "Paper's original weighted fusion f1-f4 (Eq.7 operating point)", "AUC": P.ms(rf_.auc),
                 "sens@5%": P.ms(rf_.sens, True), "winFPR@5": P.ms(rf_.fpr_win, True)})
    T1 = pd.DataFrame(rows).set_index("model"); T1.to_csv(os.path.join(a.out, "item1_baselines.csv")); print(T1.T.to_string(), flush=True)

    # ---- item 2a: bootstrap CIs over subjects
    ci = {}
    for name, s_ in sc.items():
        ci[name] = {"AUC_CI": auc_ci(s_), **{f"@{k}%": boot(s_, k) for k in TGT}}
    json.dump(ci, open(os.path.join(a.out, "item2a_bootstrap_ci.json"), "w"), indent=1); print(json.dumps(ci, indent=1), flush=True)

    # ---- items 3,4,5,8
    t3 = []
    for name, s_ in sc.items():
        for k in TGT:
            ph, n, h = alarms_per_hour(s_, k, 5.0); ph10, _, _ = alarms_per_hour(s_, k, 10.0)
            t3.append(dict(model=name, target=f"{k}%", alarms_per_h_lock5s=round(ph, 1), alarms_per_h_lock10s=round(ph10, 1), adl_hours=round(h, 2)))
    pd.DataFrame(t3).to_csv(os.path.join(a.out, "item3_alarms_per_hour.csv"), index=False); print(pd.DataFrame(t3).to_string(index=False), flush=True)

    k = 5; t4, t5, t8 = [], [], []
    for name, s_ in sc.items():
        s_ = s_.assign(ab=s_.score > s_[f"thr{k}"])
        for act, g in s_[s_.adl == 1].groupby("act"):
            t4.append(dict(model=name, act=act, activity=ACT.get(act, ""), trials=g.trial.nunique(), windows=len(g),
                           window_alarm_rate=g.ab.mean(), trial_alarm_rate=g[g.ab].trial.nunique() / g.trial.nunique()))
        pos = s_[s_.pos == 1]
        for act, g in pos.groupby("act"):
            det = g.groupby("trial").ab.max(); wt = g[g.ab].groupby("trial").dt.max()
            t5.append(dict(model=name, act=act, fall=ACT.get(act, ""), trials=len(det), sensitivity=det.mean(), warn_s=wt.mean()))
        for grp, g in s_[s_.adl == 1].assign(group=lambda x: np.where(x.subject.str.startswith("SE"), "elderly (SE)", "young adults (SA)")).groupby("group"):
            t8.append(dict(model=name, group=grp, subjects=g.subject.nunique(), adl_trials=g.trial.nunique(), window_FPR=g.ab.mean(),
                           ADL_trial_false_alarm=g[g.ab].trial.nunique() / g.trial.nunique()))
    for nm, t in (("item4_false_alarms_by_adl", t4), ("item5_sensitivity_by_fall_type", t5), ("item8_young_vs_elderly", t8)):
        pd.DataFrame(t).to_csv(os.path.join(a.out, nm + ".csv"), index=False); print("==", nm); print(pd.DataFrame(t).round(3).to_string(index=False), flush=True)

    # ---- item 2b: locked hold-out subjects (decided once, seed 42)
    rng = np.random.default_rng(42); fs = sorted(df[df.is_fall == 1].subject.unique()); nf = sorted(set(df.subject) - set(fs))
    hold = list(rng.choice(fs, 5, replace=False)) + (list(rng.choice(nf, 3, replace=False)) if len(nf) >= 3 else [])
    dev, te = df[~df.subject.isin(hold)], df[df.subject.isin(hold)]; te_ev = te[(te.pos == 1) | (te.adl == 1)]
    H = []
    for name, (kind, cols) in {**{"B0 RF baseline (8 feat)": ("rf", P.L0)}, **{n: v for n, v in CFG.items()},
                               "Logistic regression (22 feat)": ("lr", ALL), "Rule: free-fall": ("rule_ff", [8])}.items():
        P.MODEL = kind; m = P.fit_rf(dev, cols); s_ = P.score_rf(m, te_ev); row = dict(model=name, AUC=round(roc_auc_score(te_ev.pos, s_), 3))
        for tg, th in m["thr_fpr"].items():
            mm = P.op_metrics(s_, te_ev.trial.values, te_ev.pos.values == 1, te_ev.adl.values == 1, te_ev.dt.values, th); kk = int(round(tg * 100))
            row.update({f"sens@{kk}": round(mm["sens"], 3), f"winFPR@{kk}": round(mm["fpr_win"], 3), f"ADLtrialFA@{kk}": round(mm["fpr_trial"], 3),
                        f"n_fall@{kk}": mm["n_fall"]})
        H.append(row); print(row, flush=True)
    Hd = pd.DataFrame(H); Hd.to_csv(os.path.join(a.out, "item2b_locked_holdout.csv"), index=False)
    json.dump(dict(holdout_subjects=hold, n_dev_subjects=int(dev.subject.nunique())), open(os.path.join(a.out, "holdout_subjects.json"), "w"))

    # ---- item 6: permutation importance (5-fold, HGB all features, AUC drop)
    from sklearn.inspection import permutation_importance
    P.MODEL = "hgb"; imp = []
    for k_, (ia, ib) in enumerate(GroupKFold(5).split(df, groups=df.subject.values)):
        tr_, te_ = df.iloc[ia], df.iloc[ib]; ev_ = te_[(te_.pos == 1) | (te_.adl == 1)]
        if ev_.pos.sum() == 0: continue
        m = _orig_rf_fit(tr_, ALL); X = ev_[P.NAMES].values; y = ev_.pos.values
        r = permutation_importance(m, X, y, scoring="roc_auc", n_repeats=3, random_state=0)
        imp.append(r.importances_mean); print("   perm fold", k_, flush=True)
    PI = pd.DataFrame({"feature": P.NAMES, "AUC_drop_mean": np.mean(imp, 0), "AUC_drop_sd": np.std(imp, 0)}).sort_values("AUC_drop_mean", ascending=False)
    PI.to_csv(os.path.join(a.out, "item6_permutation_importance.csv"), index=False); print(PI.round(4).to_string(index=False), flush=True)

    # ---- item 7: deployability (CPU timing on THIS machine; not an embedded-device measurement)
    P.MODEL = "hgb"; mh = _orig_rf_fit(df, ALL); P.MODEL = "rf"; mr = _orig_rf_fit(df, P.L0 + P.L1)
    X = df[P.NAMES].values; D = {}
    for nm, m, cols in (("HGB all 22", mh, ALL), ("RF lever1 15", mr, P.L0 + P.L1)):
        Xc = df[[P.NAMES[c] for c in cols]].values
        t0 = time.perf_counter(); [m.predict_proba(Xc[i:i + 1]) for i in range(200)]; t1 = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter(); m.predict_proba(Xc[:5000]); tb = (time.perf_counter() - t0) / 5000
        D[nm] = dict(model_size_MB=round(len(pickle.dumps(m)) / 1e6, 2), ms_per_window_single=round(t1 * 1e3, 3), us_per_window_batch=round(tb * 1e6, 1))
    if a.sisfall:
        it = [i for i in P.list_sisfall(a.sisfall) if i["act"] == "D01"][0]; acc, gyr, mma = P.load_sisfall_file(it["path"])
        t0 = time.perf_counter(); F, T, _ = P.trial_windows(acc, gyr, mma); D["feature_extraction_ms_per_window"] = round((time.perf_counter() - t0) / len(F) * 1e3, 3)
    json.dump(D, open(os.path.join(a.out, "item7_deployability.json"), "w"), indent=1); print(json.dumps(D, indent=1), flush=True)
    print("ALL DONE")

if __name__ == "__main__":
    main()
