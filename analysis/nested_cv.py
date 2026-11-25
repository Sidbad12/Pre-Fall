import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Nested subject-wise CV on SisFall. Outer: same GroupKFold(10) as the main analysis. Inner: GroupKFold(3) on the outer-training subjects
selects among 5 candidate models (fixed in advance) by inner out-of-fold AUC; thresholds from the chosen candidate's inner OOF ADL quantiles."""
import numpy as np, pandas as pd, os, json, time, pickle, warnings
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import prefall_v3c as P
df=pd.read_pickle("real_v5/windows_cache.pkl"); L=P.L0; ALL=P.L0+P.L1+P.L3+P.L4
CANDS={
 "RF_f1-f15":("rf",P.L0+P.L1,{}),
 "RF_all22":("rf",ALL,{}),
 "HGB_all22_lr0.08_l31":("hgb",ALL,dict(learning_rate=0.08,max_leaf_nodes=31,min_samples_leaf=40)),
 "HGB_all22_lr0.05_l15":("hgb",ALL,dict(learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=100)),
 "HGB_noAcc2_lr0.08_l31":("hgb",P.L0+P.L1+P.L4,dict(learning_rate=0.08,max_leaf_nodes=31,min_samples_leaf=40)),
}
def mk(kind,hp):
    if kind=="hgb": return HistGradientBoostingClassifier(max_iter=200,l2_regularization=1.0,class_weight="balanced",random_state=0,**hp)
    return RandomForestClassifier(n_estimators=P.RF_TREES,min_samples_leaf=20,max_depth=14,class_weight="balanced_subsample",n_jobs=-1,random_state=0)
def fit(tr,kind,cols,hp):
    t=tr[(tr.pos==1)|(tr.train_neg==1)]; m=mk(kind,hp); m.fit(t[[P.NAMES[c] for c in cols]].values,t["pos"].values); return m
def sc(m,d,cols): return m.predict_proba(d[[P.NAMES[c] for c in cols]].values)[:,1]
os.makedirs("analysis_causal/nested",exist_ok=True); outs=[]; sel=[]
for k,(a,b) in enumerate(GroupKFold(10).split(df,groups=df.subject.values)):
    f=f"analysis_causal/nested/fold{k}.pkl"
    if os.path.exists(f): r=pickle.load(open(f,"rb")); outs.append(r["out"]); sel.append(r["sel"]); continue
    tr,te=df.iloc[a],df.iloc[b]
    if te.pos.sum()==0: continue
    t0=time.time(); inner={}; 
    for cn,(kind,cols,hp) in CANDS.items():
        oof=pd.Series(np.nan,index=tr.index)
        for ia,ib in GroupKFold(3).split(tr,groups=tr.subject.values):
            m=fit(tr.iloc[ia],kind,cols,hp); oof.iloc[ib]=sc(m,tr.iloc[ib],cols)
        ev=((tr.pos==1)|(tr.adl==1)).values; inner[cn]=(roc_auc_score(tr.pos.values[ev],oof.values[ev]),oof)
        print(f"fold {k} inner {cn} AUC {inner[cn][0]:.4f} ({time.time()-t0:.0f}s)",flush=True)
    best=max(inner,key=lambda c:inner[c][0]); kind,cols,hp=CANDS[best]; oof=inner[best][1]
    adl_oof=oof.values[(tr.adl==1).values]; thr={t:float(np.quantile(adl_oof,1-t)) for t in P.FPR_TARGETS}
    m=fit(tr,kind,cols,hp); d=te[["trial","subject","act","is_fall","pos","adl","dt","end"]].copy(); d["score"]=sc(m,te,cols); d["fold"]=k
    for tg,th in thr.items(): d[f"thr{int(round(tg*100))}"]=th
    d=d[(d.pos==1)|(d.adl==1)]; s=dict(fold=k,chosen=best,inner_auc={c:round(v[0],4) for c,v in inner.items()},outer_auc=round(roc_auc_score(d.pos,d.score),4))
    pickle.dump(dict(out=d,sel=s),open(f,"wb")); outs.append(d); sel.append(s); print("FOLD",k,s,flush=True)
D=pd.concat(outs); D.to_pickle("analysis_causal/oof_nested.pkl"); json.dump(sel,open("analysis_causal/item15_nested_selection.json","w"),indent=1)
print("ALL DONE")
