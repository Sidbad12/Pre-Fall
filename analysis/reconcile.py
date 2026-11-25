import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Single-definition reconciliation. Operating point O1 = per-fold threshold from the inner 3-fold out-of-fold ADL-window quantile (nominal 5%).
O2 (curve only) = pooled out-of-fold ADL-window quantile. Everything below is computed from the saved out-of-fold scores."""
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
import prefall_analysis_c as A, prefall_v3c as P
B=2000
SC={"fixed_HGB_all22":pd.read_pickle("analysis_causal/oof_C2_HGB_all.pkl"),"nested_HGB":pd.read_pickle("analysis_causal/oof_nested.pkl")}
out={}
for name,sc in SC.items():
    k=5; r={}
    pos=sc[sc.pos==1].copy(); pos["ab"]=pos.score>pos.thr5; det=pos[pos.ab].groupby("trial").dt.max(); N=pos.trial.nunique()
    adl=sc[sc.adl==1]; ab=adl.score>adl.thr5
    fm=A.fold_metrics(sc,5)
    r["n_fall_trials"]=int(N); r["n_detected"]=int(len(det)); r["n_adl_windows"]=int(len(adl)); r["n_subjects"]=int(sc.subject.nunique())
    r["trial_sens_pooled"]=round(len(det)/N,4); r["trial_sens_foldmean"]=round(fm.sens.mean(),4); r["trial_sens_fold_sd"]=round(fm.sens.std(),4)
    r["window_fpr_pooled"]=round(ab.mean(),4); r["window_fpr_foldmean"]=round(fm.fpr_win.mean(),4)
    r["adl_trial_false_alarm_share_pooled"]=round(adl[ab].trial.nunique()/adl.trial.nunique(),4)
    r["auc_pooled"]=round(roc_auc_score(sc.pos,sc.score),4); r["auc_foldmean"]=round(fm.auc.mean(),4); r["auc_fold_sd"]=round(fm.auc.std(),4)
    # window-level sensitivity (share of positive windows above threshold)
    r["window_sens_pooled"]=round(pos.ab.mean(),4)
    r["warning_mean_detected_s"]=round(det.mean(),3); r["warning_median_detected_s"]=round(det.median(),3)
    r["warning_p10_detected_s"]=round(det.quantile(.1),3); r["warning_p90_detected_s"]=round(det.quantile(.9),3)
    for t in (0.2,0.5,1.0,1.5):
        r[f"warned_ge_{t}s_share_of_all_falls"]=round((det>=t).sum()/N,4); r[f"warned_ge_{t}s_share_of_detected"]=round((det>=t).mean(),4)
    ph,n,h=A.alarms_per_hour(sc,5,5.0); ph10,_,_=A.alarms_per_hour(sc,5,10.0)
    r["alarms_per_h_lock5s"]=round(ph,1); r["alarms_per_h_lock10s"]=round(ph10,1); r["adl_hours"]=round(h,2)
    r["boot"]=A.boot(sc,5,B=B,seed=0); r["auc_CI"]=A.auc_ci(sc,B=B,seed=0); r["B"]=B
    out[name]=r; print(name,json.dumps(r,indent=1),flush=True)
# O2 curve (pooled quantile) for the fixed model, with lockout 5 s
sc=SC["fixed_HGB_all22"]; ev=sc; rows=[]
adl_s=sc[sc.adl==1].score.values
for q in (0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99,0.995,0.998):
    thr=float(np.quantile(adl_s,q)); pos=sc[sc.pos==1]; N=pos.trial.nunique(); det=pos[pos.score>thr].trial.nunique()
    s2=sc.assign(thrX=thr); n_al,hours=0,0.0
    for _,g in sc[sc.adl==1].groupby("trial"):
        g=g.sort_values("end"); hours+=g.end.max()/3600; last=-1e9
        for t_,s_ in zip(g.end.values,g.score.values):
            if s_>thr and t_-last>=5: n_al+=1; last=t_
    rows.append(dict(adl_window_quantile=q,window_fpr=round(1-q,3),alarms_per_hour=round(n_al/hours,1),fall_trial_sensitivity=round(det/N,3)))
pd.DataFrame(rows).to_csv("analysis_causal/item16_curve_pooled_quantile.csv",index=False)
json.dump(out,open("analysis_causal/item16_reconciled.json","w"),indent=1); print("ALL DONE")
