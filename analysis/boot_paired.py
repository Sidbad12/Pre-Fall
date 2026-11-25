import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, numpy as np, glob, os, json
from sklearn.metrics import roc_auc_score
D="analysis_causal"; names=["C2_HGB_all","C1_RF_lever1","base_rf_8","base_lr_22","base_svm_22","base_mlp_22","base_rule_ff_1","base_rule_tilt_1"]
S={n:pd.read_pickle(f"{D}/oof_{n}.pkl") for n in names}
ref=S["C2_HGB_all"]
def prep(df):
    d=df[(df.pos==1)|(df.adl==1)].copy(); return d.set_index(["trial","end"])
R={n:prep(d) for n,d in S.items()}
idx=R["C2_HGB_all"].index
for n in names: idx=idx.intersection(R[n].index)
print("common windows",len(idx),{n:len(R[n]) for n in names})
y=R["C2_HGB_all"].loc[idx,"pos"].values; subj=R["C2_HGB_all"].loc[idx,"subject"].values
adl=R["C2_HGB_all"].loc[idx,"adl"].values==1
sc={n:R[n].loc[idx,"score"].values for n in names}
thr={n:R[n].loc[idx,"thr5"].values for n in names}
us=np.unique(subj); ix={u:np.where(subj==u)[0] for u in us}
rng=np.random.default_rng(1); B=300
def sens5(n,i):
    # sensitivity (window-level) at pooled 5% ADL-window FPR
    s=sc[n][i]; a=adl[i]; yy=y[i]; t=np.quantile(s[a],0.95); return (s[yy==1]>t).mean()
res={n:{"dauc":[],"dsens":[]} for n in names[1:]}
for b in range(B):
    ii=np.concatenate([ix[u] for u in rng.choice(us,len(us))])
    if y[ii].min()==y[ii].max(): continue
    a0=roc_auc_score(y[ii],sc[names[0]][ii]); s0=sens5(names[0],ii)
    for n in names[1:]:
        res[n]["dauc"].append(a0-roc_auc_score(y[ii],sc[n][ii])); res[n]["dsens"].append(s0-sens5(n,ii))
rows=[]
for n in names:
    auc=roc_auc_score(y,sc[n]); s5=sens5(n,np.arange(len(y)))
    r=dict(model=n,AUC=round(auc,3),sens_at_5pct_FPR=round(s5,3))
    if n!=names[0]:
        d=np.array(res[n]["dauc"]); e=np.array(res[n]["dsens"])
        r.update(dAUC=round(roc_auc_score(y,sc[names[0]])-auc,3),dAUC_lo=round(np.percentile(d,2.5),3),dAUC_hi=round(np.percentile(d,97.5),3),p_dAUC=round(2*min((d<=0).mean(),(d>=0).mean()),4),
                 dSens=round(sens5(names[0],np.arange(len(y)))-s5,3),dSens_lo=round(np.percentile(e,2.5),3),dSens_hi=round(np.percentile(e,97.5),3))
    rows.append(r)
o=pd.DataFrame(rows); o.to_csv(f"{D}/item9_paired_bootstrap.csv",index=False); print(o.to_string())
