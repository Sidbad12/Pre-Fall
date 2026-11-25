import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, json, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import prefall_uma as U, prefall_v3c as P
dfs,_=pd.read_pickle("uma_out/uma_windows.pkl")
w,r=dfs["WAIST"],dfs["WRIST"]; B0=U.BASE
r2=r[["trial","end"]+B0].rename(columns={c:"wr_"+c for c in B0})
both=w.merge(r2,on=["trial","end"],how="inner").reset_index(drop=True)
P.NAMES=B0+["wr_"+c for c in B0]; C=U.COLS; Cw=[len(B0)+c for c in C]
cfg={"waist":C,"wrist":Cw,"waist+wrist":C+Cw}
print("common windows",len(both),"subjects",both.subject.nunique(),"fall trials",both[both.pos==1].trial.nunique(),flush=True)
out={}
for kind in ("hgb","rf"):
    P.MODEL=kind; S={}
    for n,cols in cfg.items():
        _,sc=P.run_cv(both,"rf",cols,both.subject.nunique(),None); S[n]=sc; print(kind,n,"done",flush=True)
    idx=S["waist"].index
    for n in S: idx=idx.intersection(S[n].index)
    base=S["waist"].loc[idx]; y=base.pos.values; adl=base.adl.values==1; subj=base.subject.values; trial=base.trial.values
    sc={n:S[n].loc[idx,"score"].values for n in S}
    # fall-trial max score over positive windows
    ft=pd.DataFrame({"trial":trial,"subject":subj,"pos":y}); 
    fall_tr=ft[ft.pos==1].trial.unique(); tsub=ft[ft.pos==1].groupby("trial").subject.first().loc[fall_tr].values
    tmax={n:pd.Series(sc[n][y==1]).groupby(trial[y==1]).max().loc[fall_tr].values for n in sc}
    us=np.unique(subj); ix={u:np.where(subj==u)[0] for u in us}; tix={u:np.where(tsub==u)[0] for u in us}
    def met(n,ii,ti):
        t=np.quantile(sc[n][ii][adl[ii]],0.95); return roc_auc_score(y[ii],sc[n][ii]),(sc[n][ii][y[ii]==1]>t).mean(),(tmax[n][ti]>t).mean()
    al=np.arange(len(y)); ta=np.arange(len(fall_tr))
    pt={n:met(n,al,ta) for n in sc}
    rng=np.random.default_rng(11); B=2000; R={n:[] for n in sc}
    for b in range(B):
        d=rng.choice(us,len(us)); ii=np.concatenate([ix[u] for u in d]); ti=np.concatenate([tix[u] for u in d if len(tix[u])]) 
        if y[ii].min()==y[ii].max(): continue
        for n in sc: R[n].append(met(n,ii,ti))
    R={n:np.array(v) for n,v in R.items()}
    rows=[]
    for n in sc:
        a=R[n]; rows.append(dict(kind=kind,config=n,AUC=round(pt[n][0],3),AUC_lo=round(np.percentile(a[:,0],2.5),3),AUC_hi=round(np.percentile(a[:,0],97.5),3),
          win_sens5=round(pt[n][1],3),trial_sens5=round(pt[n][2],3),trial_lo=round(np.percentile(a[:,2],2.5),3),trial_hi=round(np.percentile(a[:,2],97.5),3)))
    pairs=[("waist","wrist"),("waist+wrist","waist"),("waist+wrist","wrist")]; prow=[]
    for a_,b_ in pairs:
        for j,mn in enumerate(["dAUC","d_win_sens","d_trial_sens"]):
            d=R[a_][:,j]-R[b_][:,j]; m=len(d); p=min(1,2*min(((d<=0).sum()+1)/(m+1),((d>=0).sum()+1)/(m+1)))
            prow.append(dict(kind=kind,contrast=f"{a_} minus {b_}",metric=mn,point=round(pt[a_][j]-pt[b_][j],3),lo=round(np.percentile(d,2.5),3),hi=round(np.percentile(d,97.5),3),p_raw=round(p,4),B=m))
    pd.DataFrame(rows).to_csv(f"analysis_causal/item14_uma_{kind}_configs.csv",index=False)
    pd.DataFrame(prow).to_csv(f"analysis_causal/item14_uma_{kind}_paired.csv",index=False)
    print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(prow).to_string(),flush=True)
    out[kind]=dict(n_windows=int(len(y)),n_fall_trials=int(len(fall_tr)),n_subjects=int(len(us)))
json.dump(out,open("analysis_causal/item14_uma_meta.json","w"))
print("ALL DONE")
