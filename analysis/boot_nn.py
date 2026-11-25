import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D="analysis_causal"
h=pd.read_pickle(f"{D}/oof_C2_HGB_all.pkl")[["trial","end","subject","pos","adl","score"]]
h["k"]=h.trial.astype(int).astype(str)+"_"+h.end.round(3).astype(str)
S={"HGB":h.set_index("k")}
for n in ("cnn","gru"):
    d=pd.read_csv(f"{D}/oof_nn_{n}.csv"); d["k"]=d.trial.astype(int).astype(str)+"_"+d.end.round(3).astype(str); S[n]=d.set_index("k")
idx=S["HGB"].index
for n in ("cnn","gru"): idx=idx.intersection(S[n].index)
print("common",len(idx),{k:len(v) for k,v in S.items()})
y=S["HGB"].loc[idx,"pos"].values; adl=S["HGB"].loc[idx,"adl"].values==1; subj=S["HGB"].loc[idx,"subject"].values
sc={k:S[k].loc[idx,"score"].values for k in S}
us=np.unique(subj); ix={u:np.where(subj==u)[0] for u in us}
def s5(s,i):
    t=np.quantile(s[i][adl[i]],0.95); return (s[i][y[i]==1]>t).mean()
rng=np.random.default_rng(2); R={n:{"a":[],"s":[]} for n in ("cnn","gru")}
for _ in range(300):
    ii=np.concatenate([ix[u] for u in rng.choice(us,len(us))])
    if y[ii].min()==y[ii].max(): continue
    a0=roc_auc_score(y[ii],sc["HGB"][ii]); b0=s5(sc["HGB"],ii)
    for n in R: R[n]["a"].append(a0-roc_auc_score(y[ii],sc[n][ii])); R[n]["s"].append(b0-s5(sc[n],ii))
al=np.arange(len(y))
for n in ("HGB","cnn","gru"): print(n,"AUC",round(roc_auc_score(y,sc[n]),3),"sens5",round(s5(sc[n],al),3))
for n in R:
    a=np.array(R[n]["a"]); s=np.array(R[n]["s"])
    print(f"HGB-{n}: dAUC {roc_auc_score(y,sc['HGB'])-roc_auc_score(y,sc[n]):.3f} CI {np.percentile(a,2.5):.3f},{np.percentile(a,97.5):.3f} | dSens {s5(sc['HGB'],al)-s5(sc[n],al):.3f} CI {np.percentile(s,2.5):.3f},{np.percentile(s,97.5):.3f}")
