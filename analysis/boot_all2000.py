import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score
D="analysis_causal"; B=2000
names=["C2_HGB_all","C1_RF_lever1","base_rf_8","base_lr_22","base_svm_22","base_mlp_22","base_rule_ff_1","base_rule_tilt_1"]
def key(df): return df.trial.astype(int).astype(str)+"_"+df.end.round(3).astype(str)
S={}
for n in names:
    d=pd.read_pickle(f"{D}/oof_{n}.pkl"); d=d[(d.pos==1)|(d.adl==1)].copy(); d["k"]=key(d); S[n]=d.drop_duplicates("k").set_index("k")
for n in ("cnn","gru"):
    d=pd.read_csv(f"{D}/oof_nn_{n}.csv"); d["k"]=key(d); S[n]=d.drop_duplicates("k").set_index("k")
allm=names+["cnn","gru"]
idx=S[allm[0]].index
for n in allm: idx=idx.intersection(S[n].index)
print("common",len(idx),flush=True)
y=S[allm[0]].loc[idx,"pos"].values; adl=S[allm[0]].loc[idx,"adl"].values==1; subj=S[allm[0]].loc[idx,"subject"].values
sc={n:S[n].loc[idx,"score"].values for n in allm}
us=np.unique(subj); ix={u:np.where(subj==u)[0] for u in us}
def s5(s,i):
    t=np.quantile(s[i][adl[i]],0.95); return (s[i][y[i]==1]>t).mean()
al=np.arange(len(y)); rng=np.random.default_rng(7)
R={n:{"a":[],"s":[]} for n in allm[1:]}
for b in range(B):
    ii=np.concatenate([ix[u] for u in rng.choice(us,len(us))])
    if y[ii].min()==y[ii].max(): continue
    a0=roc_auc_score(y[ii],sc[allm[0]][ii]); b0=s5(sc[allm[0]],ii)
    for n in R: R[n]["a"].append(a0-roc_auc_score(y[ii],sc[n][ii])); R[n]["s"].append(b0-s5(sc[n],ii))
    if b%200==0: print(b,flush=True)
rows=[]; a0f=roc_auc_score(y,sc[allm[0]]); s0f=s5(sc[allm[0]],al)
for n in allm[1:]:
    a=np.array(R[n]["a"]); s=np.array(R[n]["s"]); m=len(a)
    p=2*min(((a<=0).sum()+1)/(m+1),((a>=0).sum()+1)/(m+1)); p=min(p,1.0)
    rows.append(dict(baseline=n,AUC=round(roc_auc_score(y,sc[n]),3),dAUC=round(a0f-roc_auc_score(y,sc[n]),3),dAUC_lo=round(np.percentile(a,2.5),3),dAUC_hi=round(np.percentile(a,97.5),3),p_raw=p,
      sens5=round(s5(sc[n],al),3),dSens=round(s0f-s5(sc[n],al),3),dSens_lo=round(np.percentile(s,2.5),3),dSens_hi=round(np.percentile(s,97.5),3),B=m))
o=pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True); k=len(o)
o["p_holm"]=np.minimum(1,np.maximum.accumulate([(k-i)*p for i,p in enumerate(o.p_raw)]))
o.to_csv(f"{D}/item9b_paired_bootstrap_2000_holm.csv",index=False); print(o.to_string()); print("HGB",round(a0f,3),round(s0f,3))
