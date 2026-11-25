import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, pickle, os, itertools
import prefall_v3c as P
root=os.environ.get("SISFALL_ROOT","SisFall_dataset")
S=pd.read_pickle("analysis_causal/alarm/all_scores.pkl"); S=S[(S.pos==1)|(S.adl==1)].sort_values(["trial","end"]).reset_index(drop=True)
items=P.list_sisfall(root)
cache="mag_cache.pkl"
if os.path.exists(cache): MAG=pickle.load(open(cache,"rb"))
else:
    MAG={}
    for t in S.trial.unique():
        acc,_,_=P.load_sisfall_file(items[int(t)]["path"]); MAG[int(t)]=np.linalg.norm(acc,axis=1).astype(np.float32)
    pickle.dump(MAG,open(cache,"wb"))
FS=P.FS
def feats(t,tw):
    m=MAG[t]; return m
def alarm_feats(trial,t_alarm,tw):
    m=MAG[trial]; i0=int(round(t_alarm*FS)); i1=min(len(m),i0+int(tw*FS))
    if i1-i0<5: return 0.0,np.inf,0
    seg=m[i0:i1]; j=int(np.argmax(seg)); pk=float(seg[j]); tp=i0+j
    a=tp+int(0.6*FS); b=min(len(m),tp+int(2.6*FS))
    if b-a<int(1.0*FS): return pk,np.inf,0         # not enough data -> conservative (no stillness check passes? treated as pass)
    return pk,float(np.std(m[a:b])),1
def stage1_alarms(df,thr,lock=5.0):
    rows=[]
    for tr,g in df.groupby("trial"):
        last=-1e9
        for t_,s_ in zip(g.end.values,g.s_base.values):
            if s_>thr and t_-last>=lock: rows.append((tr,t_)); last=t_
    return pd.DataFrame(rows,columns=["trial","t"])
TW=(1.5,2.5); GS=(1.8,2.2,2.6,3.0,3.5,4.0); SS=(0.05,0.1,0.2,0.35,9e9)
meta=S.groupby("trial").agg(subject=("subject","first"),act=("act","first"),is_fall=("is_fall","first"),adl=("adl","max"),fold=("fold","first"),dur=("end","max"))
meta["hours"]=np.where(meta.adl==1,meta.dur/3600,0)
def prep(df,thr):
    A=stage1_alarms(df,thr)
    if len(A)==0: return A
    for tw in TW:
        f=[alarm_feats(int(a),t,tw) for a,t in zip(A.trial,A.t)]
        A[f"pk{tw}"]=[x[0] for x in f]; A[f"sd{tw}"]=[x[1] for x in f]
    A["fold"]=A.trial.map(meta.fold); A["is_fall"]=A.trial.map(meta.is_fall); return A
def evaluate(A,fold_mask_trials,tw,G,s0):
    ok=(A[f"pk{tw}"]>=G)&(A[f"sd{tw}"]<s0)
    C=A[ok]; ft=meta[(meta.is_fall==1)&fold_mask_trials]; at=meta[(meta.adl==1)&fold_mask_trials]
    det=C[(C.is_fall==1)&C.trial.isin(ft.index)].trial.nunique() if len(C) else 0
    fa=C[C.is_fall==0]; fa=fa[fa.trial.isin(at.index)]
    return det,len(ft),len(fa),at.hours.sum(),fa.trial.nunique(),len(at)
folds=sorted(meta.fold.dropna().unique()); TARG=(0.90,0.95,0.98); SENS_MIN=0.90
out=[];base=[]
for k in folds:
    trk=meta.fold!=k; tek=meta.fold==k
    sel=None
    cand={}
    for tg in TARG:
        trn=S[S.fold!=k]; tm=trn[trn.pos==1].groupby("trial").s_base.max()
        thr=float(np.quantile(tm.values,1-tg))
        A=prep(S,thr); cand[tg]=(thr,A)
        for tw,G,s0 in itertools.product(TW,GS,SS):
            det,nf,fa,h,fat,nat=evaluate(A,trk,tw,G,s0)
            if det/nf>=SENS_MIN and (sel is None or fa/h<sel[0]): sel=(fa/h,tg,tw,G,s0)
    if sel is None: sel=(None,TARG[-1],1.5,1.8,9e9)
    _,tg,tw,G,s0=sel; thr,A=cand[tg]
    det,nf,fa,h,fat,nat=evaluate(A,tek,tw,G,s0)
    # stage-1 alone at same thr on held-out
    A1=A; d1=A1[(A1.is_fall==1)&A1.trial.isin(meta[tek].index)].trial.nunique()
    f1=len(A1[(A1.is_fall==0)&A1.trial.isin(meta[(tek)&(meta.adl==1)].index)])
    out.append(dict(fold=k,tg=tg,tw=tw,G=G,s0=s0,det=det,nf=nf,fa=fa,hours=h,adl_trials_alerted=fat,n_adl=nat,s1_det=d1,s1_fa=f1))
    print(out[-1],flush=True)
R=pd.DataFrame(out); R.to_csv("analysis_causal/item22_two_stage_folds.csv",index=False)
tot=R[["det","nf","fa","hours","adl_trials_alerted","n_adl","s1_det","s1_fa"]].sum()
print("TWO-STAGE: sens %.3f  confirmed alerts/h %.2f  ADL trials alerted %.3f"%(tot.det/tot.nf,tot.fa/tot.hours,tot.adl_trials_alerted/tot.n_adl))
print("STAGE-1 (same thr, arm only): sens %.3f alarms/h %.1f"%(tot.s1_det/tot.nf,tot.s1_fa/tot.hours))
