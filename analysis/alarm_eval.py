import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
D=pd.read_pickle("analysis_causal/alarm/all_scores.pkl").sort_values(["trial","end"]).reset_index(drop=True)
for v in ("base","mined"):
    D[f"s_{v}_m2"]=D.groupby("trial")[f"s_{v}"].transform(lambda x:x.rolling(2,min_periods=1).mean())
    D[f"s_{v}_m3"]=D.groupby("trial")[f"s_{v}"].transform(lambda x:x.rolling(3,min_periods=1).mean())
V=["s_base","s_base_m2","s_base_m3","s_mined","s_mined_m2","s_mined_m3"]
E=D[(D.pos==1)|(D.adl==1)].copy(); adl=E.adl.values==1; y=E.pos.values
def alarms_h(df,col,thr,lock=5.0):
    n,h=0,0.0
    for _,g in df[df.adl==1].groupby("trial"):
        h+=g.end.max()/3600; last=-1e9
        for t_,s_ in zip(g.end.values,g[col].values):
            if s_>thr and t_-last>=lock: n+=1; last=t_
    return n/h
pos=E[E.pos==1]; fall_tr=pos.trial.unique(); tsub=pos.groupby("trial").subject.first().loc[fall_tr].values
rows=[]
for v in V:
    s=E[v].values; auc=roc_auc_score(y,s)
    tm=pd.Series(s[y==1]).groupby(E.trial.values[y==1]).max().loc[fall_tr].values
    for q in (0.995,0.99,0.98,0.97,0.95):
        thr=float(np.quantile(s[adl],q))
        rows.append(dict(variant=v,AUC=round(auc,4),adl_window_quantile=q,window_fpr=round(1-q,3),alarms_per_h=round(alarms_h(E,v,thr),1),
            trial_sens=round((tm>thr).mean(),4),window_sens=round((s[y==1]>thr).mean(),4)))
R=pd.DataFrame(rows); R.to_csv("analysis_causal/item19_alarm_variants.csv",index=False); print(R.to_string())
# paired subject bootstrap for trial-sensitivity at matched window FPR vs base
subj=E.subject.values; us=np.unique(subj); ix={u:np.where(subj==u)[0] for u in us}; tix={u:np.where(tsub==u)[0] for u in us}
tmax={v:pd.Series(E[v].values[y==1]).groupby(E.trial.values[y==1]).max().loc[fall_tr].values for v in V}
rng=np.random.default_rng(3); B=1000; out={}
for q in (0.99,0.98,0.95):
    d={v:[] for v in V[1:]}
    for _ in range(B):
        c=rng.choice(us,len(us)); ii=np.concatenate([ix[u] for u in c]); ti=np.concatenate([tix[u] for u in c if len(tix[u])]); a=adl[ii]
        def ts(v): t=np.quantile(E[v].values[ii][a],q); return (tmax[v][ti]>t).mean()
        b0=ts("s_base")
        for v in V[1:]: d[v].append(ts(v)-b0)
    for v in V[1:]:
        x=np.array(d[v]); m=len(x); p=min(1,2*min(((x<=0).sum()+1)/(m+1),((x>=0).sum()+1)/(m+1)))
        out[f"{v}@fpr{round((1-q)*100)}%"]=dict(diff=round(float(np.mean(x)),3),lo=round(float(np.percentile(x,2.5)),3),hi=round(float(np.percentile(x,97.5)),3),p_raw=round(p,4))
json.dump(out,open("analysis_causal/item19_alarm_boot.json","w"),indent=1)
for k,v in out.items(): print(k,v)
# elderly vs young false alarms at base pooled 5% threshold (per variant own 5% pooled threshold)
E["grp"]=np.where(E.subject.str.startswith("SE"),"elderly","adult"); g=[]
for v in V:
    thr=float(np.quantile(E[v].values[adl],0.95))
    for grp,sub in E.groupby("grp"):
        a=sub[sub.adl==1]; g.append(dict(variant=v,group=grp,n_subjects=sub.subject.nunique(),adl_window_fpr=round((a[v]>thr).mean(),4),alarms_per_h=round(alarms_h(sub,v,thr),1)))
G=pd.DataFrame(g); G.to_csv("analysis_causal/item19_alarm_by_age.csv",index=False); print(G.to_string())
