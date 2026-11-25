import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, pickle, itertools
MAG=pickle.load(open("mag_cache.pkl","rb")); FS=200.0
S=pd.read_pickle("analysis_causal/alarm/all_scores.pkl")
meta=S.groupby("trial").agg(is_fall=("is_fall","first"),adl=("adl","max"),fold=("fold","first"),act=("act","first"))
meta["hours"]=[len(MAG[int(t)])/FS/3600 if a==1 else 0 for t,a in zip(meta.index,meta.adl)]
meta=meta[(meta.is_fall==1)|(meta.adl==1)]
GS=(2.2,2.6,3.0,3.5,4.0,5.0); SS=(0.05,0.1,0.2,0.35,9e9)
ev={}
for t in meta.index:
    m=MAG[int(t)]; idx=np.where(m>=min(GS))[0]; evs=[]; last=-1e9
    for i in idx:
        if i-last>=5*FS:
            j=i+int(1.0*FS); seg=m[i:j]; tp=i+int(np.argmax(seg)); last=i
            a=tp+int(0.6*FS); b=min(len(m),tp+int(2.6*FS))
            sd=np.std(m[a:b]) if b-a>=int(1.0*FS) else np.inf
            evs.append((float(m[tp]),float(sd)))
    ev[t]=evs
def ev_count(t,G,s0): return sum(1 for pk,sd in ev[t] if pk>=G and sd<s0)
folds=sorted(meta.fold.unique()); rows=[]
for k in folds:
    best=None
    trn=meta[meta.fold!=k]
    for G,s0 in itertools.product(GS,SS):
        det=np.mean([ev_count(t,G,s0)>0 for t in trn[trn.is_fall==1].index])
        fa=sum(ev_count(t,G,s0) for t in trn[trn.adl==1].index)/trn.hours.sum()
        if det>=0.90 and (best is None or fa<best[0]): best=(fa,G,s0)
    if best is None: best=(None,GS[0],SS[-1])
    _,G,s0=best; te=meta[meta.fold==k]
    rows.append(dict(fold=k,G=G,s0=s0,det=sum(ev_count(t,G,s0)>0 for t in te[te.is_fall==1].index),nf=int((te.is_fall==1).sum()),
                     fa=sum(ev_count(t,G,s0) for t in te[te.adl==1].index),hours=te.hours.sum(),
                     alerted=sum(ev_count(t,G,s0)>0 for t in te[te.adl==1].index),n_adl=int((te.adl==1).sum())))
R=pd.DataFrame(rows); print(R); t=R.sum()
print("STAGE-2 ALONE: sens %.3f alerts/h %.2f ADL trials alerted %.3f"%(t.det/t.nf,t.fa/t.hours,t.alerted/t.n_adl))
R.to_csv("analysis_causal/item22_stage2_alone.csv",index=False)
print("--- stage-2 alone, pooled fixed grid (descriptive, not nested) ---")
F=meta[meta.is_fall==1].index; A=meta[meta.adl==1].index; H=meta.hours.sum()
res=[]
for G,s0 in itertools.product(GS,SS):
    res.append((G,s0,np.mean([ev_count(t,G,s0)>0 for t in F]),sum(ev_count(t,G,s0) for t in A)/H))
r=pd.DataFrame(res,columns=["G","s0","sens","alerts_h"]); print(r[(r.sens>0.84)&(r.sens<0.93)].sort_values("sens").to_string())
