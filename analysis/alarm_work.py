import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Alarm-rate experiments (SisFall, subject-wise 10 folds). All variants use only training-subject data for fitting/mining.
V0 baseline HGB; V2 hard-negative mining (ADL/negative windows whose inner-OOF score is in the top 10% of ADL scores get weight 5, then refit);
smoothing = causal mean of the last k window scores within a trial. Scores saved for ALL test windows so smoothing is contiguous."""
import numpy as np, pandas as pd, os, pickle, time, warnings
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
import prefall_v3c as P
df=pd.read_pickle("real_v5/windows_cache.pkl"); ALL=P.L0+P.L1+P.L3+P.L4; X=[P.NAMES[c] for c in ALL]
os.makedirs("analysis_causal/alarm",exist_ok=True)
def mk(): return HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=1.0,class_weight="balanced",random_state=0)
def fit(tr,w=None):
    t=tr[(tr.pos==1)|(tr.train_neg==1)]; m=mk(); m.fit(t[X].values,t["pos"].values,sample_weight=None if w is None else w.loc[t.index].values); return m
outs=[]
for k,(a,b) in enumerate(GroupKFold(10).split(df,groups=df.subject.values)):
    f=f"analysis_causal/alarm/fold{k}.pkl"
    if os.path.exists(f): outs.append(pickle.load(open(f,"rb"))); continue
    tr,te=df.iloc[a],df.iloc[b]
    if te.pos.sum()==0: continue
    t0=time.time(); oof=pd.Series(np.nan,index=tr.index)
    for ia,ib in GroupKFold(3).split(tr,groups=tr.subject.values):
        m=fit(tr.iloc[ia]); oof.iloc[ib]=m.predict_proba(tr.iloc[ib][X].values)[:,1]
    neg=((tr.adl==1)|(tr.train_neg==1)).values; hard=neg&(oof.values>=np.quantile(oof.values[(tr.adl==1).values],0.90))
    w=pd.Series(1.0,index=tr.index); w[hard]=5.0
    m0=fit(tr); m2=fit(tr,w)
    d=te[["trial","subject","act","is_fall","pos","adl","dt","end"]].copy()
    d["s_base"]=m0.predict_proba(te[X].values)[:,1]; d["s_mined"]=m2.predict_proba(te[X].values)[:,1]; d["fold"]=k
    pickle.dump(d,open(f,"wb")); outs.append(d); print("fold",k,"done",round(time.time()-t0),"s; hard negs",int(hard.sum()),flush=True)
D=pd.concat(outs); D.to_pickle("analysis_causal/alarm/all_scores.pkl"); print("ALL DONE")
