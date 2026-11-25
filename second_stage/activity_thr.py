import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
exec(open("activity_gate.py").read().split("GRID=")[0])
def opt_thr(tr,str_,target=TARGET,iters=4):
    K=np.unique(str_); s=tr.s_base.values; adl=tr.adl.values==1; pos=tr.pos.values==1
    qs=np.quantile(s[adl],np.linspace(0.8,0.999,60))
    thr={k:np.quantile(s[adl],0.95) for k in K}
    tid=tr.trial.values
    def sens(th):
        t=np.array([th[k] for k in str_]); hit=(s>t)&pos
        return pd.Series(hit[pos]).groupby(tid[pos]).max().mean()
    def fpr(th):
        t=np.array([th[k] for k in str_]); return ((s>t)&adl).sum()
    for _ in range(iters):
        for k in K:
            best=None
            for q in qs:
                th=dict(thr); th[k]=q
                if sens(th)>=target:
                    f=fpr(th)
                    if best is None or f<best[0]: best=(f,q)
            if best: thr[k]=best[1]
    # fine tune to hit target exactly via global shift
    return thr
rows=[]
for k in sorted(E.fold.unique()):
    tr=E[E.fold!=k].reset_index(drop=True); te=E[E.fold==k].reset_index(drop=True)
    str_,ste=states(tr,te)
    thr=opt_thr(tr,str_)
    t_te=np.array([thr[x] for x in ste]); eff=np.where(te.s_base.values>t_te,te.s_base.values,-1.0)
    tm=trial_max(te,eff); hrs=te[te.adl==1].groupby("trial").end.max().sum()/3600
    rows.append(dict(det=int((tm>0).sum()),n=len(tm),alarms=alarms_h(te,eff,0)*hrs,hours=hrs,thr=str({kk:round(v,3) for kk,v in thr.items()})))
R=pd.DataFrame(rows); print(R[["det","n","alarms","hours","thr"]]); print("sens",R.det.sum()/R.n.sum(),"alarms/h",R.alarms.sum()/R.hours.sum())
