import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, itertools, json
from sklearn.cluster import KMeans
S=pd.read_pickle("analysis_causal/alarm/all_scores.pkl")
W=pd.read_pickle("real_v5/windows_cache.pkl")
FE=["act_intensity","gait_band_ratio","dom_freq","vvel_change_last05","tilt_delta_last05","jerk_last05","gyro_rms_last05","min_mag_1s","freefall_frac_1s"]
D=S.merge(W[["trial","end"]+FE],on=["trial","end"],how="left")
assert len(D)==len(S) and D[FE].notna().all(axis=1).mean()>0.99
D=D.fillna(D[FE].median()); D=D.sort_values(["trial","end"]).reset_index(drop=True)
D["vv"]=D.vvel_change_last05.abs()
GF=["vv","tilt_delta_last05","jerk_last05","gyro_rms_last05"]
SF=["act_intensity","gait_band_ratio","dom_freq"]
E=D[(D.pos==1)|(D.adl==1)].reset_index(drop=True)
TARGET=0.90
def alarms_h(df,eff,thr,lock=5.0):
    n,h=0,0.0
    for _,g in df[df.adl==1].groupby("trial"):
        h+=g.end.max()/3600; last=-1e9
        for t_,s_ in zip(g.end.values,eff[g.index.values]):
            if s_>thr and t_-last>=lock: n+=1; last=t_
    return n/h
def trial_max(df,eff):
    p=df[df.pos==1]; return pd.Series(eff[p.index.values]).groupby(p.trial.values).max()
def states(tr,te,k=4):
    X=lambda d:np.log1p(d[SF].values.clip(0)); 
    mu,sd=X(tr).mean(0),X(tr).std(0)+1e-9
    km=KMeans(k,n_init=5,random_state=0).fit((X(tr)-mu)/sd)
    # order clusters by mean intensity
    order=np.argsort(km.cluster_centers_[:,0]); rk={c:i for i,c in enumerate(order)}
    f=lambda d:np.array([rk[c] for c in km.predict((X(d)-mu)/sd)])
    return f(tr),f(te)
def make_eff(variant,tr,te,str_,ste,p=None,m=None):
    s_tr,s_te=tr.s_base.values,te.s_base.values
    if variant=="base": return s_tr,s_te
    if variant=="qnorm":  # state-conditional ADL percentile of score
        out=[]
        for st,s,sa in ((str_,s_tr,tr),(ste,s_te,te)):
            e=np.zeros(len(s))
            for k in np.unique(str_):
                ref=np.sort(tr.s_base.values[(str_==k)&(tr.adl.values==1)])
                msk=st==k; e[msk]=np.searchsorted(ref,s[msk])/max(len(ref),1)
            out.append(e)
        return out
    if variant=="gate":
        def gate(d,st):
            cnt=np.zeros(len(d))
            for f in GF:
                th=np.zeros(len(d))
                for k in np.unique(str_):
                    ref=tr[f].values[(str_==k)&(tr.adl.values==1)]
                    th[st==k]=np.quantile(ref,p)
                cnt+= d[f].values>=th
            return cnt>=m
        return [np.where(gate(tr,str_),s_tr,-1.0),np.where(gate(te,ste),s_te,-1.0)]
GRID=[(p,m) for p in (0.5,0.7,0.85) for m in (1,2,3)]
rows=[];sel=[]
for k in sorted(E.fold.unique()):
    tr=E[E.fold!=k].reset_index(drop=True); te=E[E.fold==k].reset_index(drop=True)
    str_,ste=states(tr,te)
    cands={"base":make_eff("base",tr,te,str_,ste),"qnorm":make_eff("qnorm",tr,te,str_,ste)}
    for p,m in GRID: cands[f"gate_p{p}_m{m}"]=make_eff("gate",tr,te,str_,ste,p,m)
    # pick best gate on training folds: lowest train alarms/h at target sens
    res={}
    for name,(etr,ete) in cands.items():
        tm=trial_max(tr,etr); thr=np.quantile(tm.values,1-TARGET)
        res[name]=(thr,alarms_h(tr,etr,thr))
    gbest=min([n for n in res if n.startswith("gate")],key=lambda n:res[n][1]); sel.append(gbest)
    for name in ("base","qnorm",gbest):
        etr,ete=cands[name]; thr=res[name][0]
        tm=trial_max(te,ete); ah=alarms_h(te,ete,thr)
        n_tr=te[te.pos==1].trial.nunique()
        nalarm=ah*te[te.adl==1].groupby("trial").end.max().sum()/3600
        hrs=te[te.adl==1].groupby("trial").end.max().sum()/3600
        rows.append(dict(fold=k,variant="gate_selected" if name==gbest else name,det=int((tm>thr).sum()),n_falls=len(tm),alarms=nalarm,hours=hrs,
                         win_fpr=float((ete[te.adl.values==1]>thr).mean())))
R=pd.DataFrame(rows).groupby("variant").agg(det=("det","sum"),n=("n_falls","sum"),alarms=("alarms","sum"),hours=("hours","sum"),win_fpr=("win_fpr","mean"))
R["sens"]=R.det/R.n; R["alarms_per_h"]=R.alarms/R.hours
print(R.round(3)); print(pd.Series(sel).value_counts())
R.to_csv("analysis_causal/item21_activity_gate.csv")
# state mapping on full data
st_all,_=states(E,E.iloc[:5]); E["state"]=st_all
mp=pd.crosstab(E[E.adl==1].act,E[E.adl==1].state,normalize="index").round(2); mp.to_csv("analysis_causal/item21_state_map.csv"); print(mp.to_string())
print(E[E.adl==1].groupby("state")[SF].median().round(2))
E.to_pickle("E_state.pkl")
