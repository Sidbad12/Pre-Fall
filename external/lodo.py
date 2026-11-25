import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, numpy as np, pickle, itertools, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
names=pickle.load(open("ltmm_pack/ltmm_model.pkl","rb"))["names"]
sf=pd.read_pickle("real_v5/windows_cache.pkl"); kf=pd.read_pickle("cross_out/kfall_windows.pkl")
fa=pd.read_pickle(os.environ.get("FALLALLD_FEATS","fallalld_feats.pkl"))
fa=fa.rename(columns={"fall":"is_fall"}); fa["pos"]=fa.is_fall; fa["adl"]=1-fa.is_fall; fa["train_neg"]=fa.adl
D={}
for n,d in (("SisFall",sf),("KFall",kf),("FallAllD",fa)):
    d=d[["trial","subject","pos","adl","train_neg"]+names].copy().reset_index(drop=True); d["ds"]=n; D[n]=d
def adl_norm(d):   # per-dataset robust standardisation with that dataset's own ADL windows (= recalibration on a sample of the wearer's/device's normal activity)
    a=d[d.adl==1][names]; med=a.median(); iqr=(a.quantile(.75)-a.quantile(.25)).replace(0,1e-6)
    e=d.copy(); e[names]=(d[names]-med)/iqr; return e
DN={k:adl_norm(v) for k,v in D.items()}
def fit(parts):
    X=[];y=[];w=[]
    for d in parts:
        t=d[(d.pos==1)|(d.train_neg==1)]; yy=t.pos.values; ww=np.where(yy==1,0.5/max(yy.sum(),1),0.5/max((1-yy).sum(),1))   # class-balanced, each dataset total weight 1
        X.append(t[names].values);y.append(yy);w.append(ww)
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=1.0,random_state=0)
    W=np.concatenate(w); W=W*len(W)/W.sum()   # mean weight 1, so the L2 term keeps its usual scale
    m.fit(np.vstack(X),np.concatenate(y),sample_weight=W); return m
def evaluate(m,d,rng):
    e=d[(d.pos==1)|(d.adl==1)].reset_index(drop=True); s=m.predict_proba(e[names].values)[:,1]; y=e.pos.values
    auc=roc_auc_score(y,s); thr=np.quantile(s[e.adl.values==1],0.95)
    ft=e[e.pos==1].assign(hit=(s[e.pos.values==1]>thr)).groupby("trial").hit.max().mean()
    subs=e.subject.values; us=np.unique(subs); idx={u:np.where(subs==u)[0] for u in us}; b=[]
    for _ in range(300):
        ii=np.concatenate([idx[u] for u in rng.choice(us,len(us))])
        if y[ii].min()!=y[ii].max(): b.append(roc_auc_score(y[ii],s[ii]))
    return auc,np.percentile(b,2.5),np.percentile(b,97.5),ft
rows=[]; rng=np.random.default_rng(0)
for tgt in D:
    others=[k for k in D if k!=tgt]
    for norm in (False,True):
        src=DN if norm else D
        for tr in [[o] for o in others]+[others]:
            m=fit([src[k] for k in tr]); auc,lo,hi,ft=evaluate(m,src[tgt],rng)
            rows.append(dict(test=tgt,train="+".join(tr),adl_normalised=norm,AUC=round(auc,3),lo=round(lo,3),hi=round(hi,3),fall_sens_at_5pct=round(ft,3))); print(rows[-1],flush=True)
R=pd.DataFrame(rows); R.to_csv("analysis_causal/item23_lodo.csv",index=False); print(R.to_string())
