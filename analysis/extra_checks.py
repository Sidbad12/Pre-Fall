import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, json
import prefall_v3c as P
D="analysis_causal"
h=pd.read_pickle(f"{D}/oof_C2_HGB_all.pkl").sort_values(["trial","end"]).reset_index(drop=True)
e=h[(h.pos==1)|(h.adl==1)]
out={}
# (a) warning time percentiles at 5% operating point (per fall trial: impact - end of FIRST alarmed positive window)
t=np.quantile(e[e.adl==1].score,0.95)
f=e[e.pos==1]; al=f[f.score>t]; first=al.groupby("trial").dt.max()      # dt = impact - end ; largest dt = earliest alarm
ntr=f.trial.nunique(); w=first.values
out["warning"]={"n_fall_trials":int(ntr),"detected_trials":int(len(w)),"median_s":float(np.median(w)),"p10_s":float(np.percentile(w,10)),"p90_s":float(np.percentile(w,90)),
  **{f"share_of_all_falls_warned_ge_{int(x*1000)}ms":float((w>=x).sum()/ntr) for x in (0.2,0.5,1.0)}}
# (b) sensitivity (trial-level) vs alarms/hour curve
rows=[]
adl=h[h.adl==1]
for q in (0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99,0.995,0.998):
    thr=np.quantile(e[e.adl==1].score,q); n_al=0; hrs=0.0
    for _,g in adl.groupby("trial"):
        hrs+=g.end.max()/3600; last=-1e9
        for tt,ss in zip(g.end.values,g.score.values):
            if ss>thr and tt-last>=5.0: n_al+=1; last=tt
    ft=f.assign(a=f.score>thr).groupby("trial").a.any().mean()
    rows.append(dict(adl_window_quantile=q,window_fpr=round(1-q,4),alarms_per_hour=round(n_al/hrs,1),fall_trial_sensitivity=round(float(ft),3)))
pd.DataFrame(rows).to_csv(f"{D}/item11_sens_vs_alarms_per_hour.csv",index=False); print(pd.DataFrame(rows).to_string())
json.dump(out,open(f"{D}/item12_warning_percentiles.json","w"),indent=1); print(json.dumps(out,indent=1))
