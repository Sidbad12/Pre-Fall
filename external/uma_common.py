import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys; sys.path.insert(0,'.')
import pandas as pd, prefall_uma as U, prefall_v3c as P
dfs,_=pd.read_pickle('uma_out/uma_windows.pkl')
w,r=dfs['WAIST'],dfs['WRIST']
keys=set(zip(w.trial,w.end))&set(zip(r.trial,r.end))
rows=[]
for kind in ('rf','hgb'):
    P.MODEL=kind
    for name,d in (('WAIST (common windows)',w),('WRIST (common windows)',r)):
        m=[k in keys for k in zip(d.trial,d.end)]; dd=d[m]
        _,sc=P.run_cv(dd,'rf',U.COLS,dd.subject.nunique(),None); sc['end']=dd.loc[sc.index,'end']; U.pooled_eval(sc,name,kind,rows)
pd.DataFrame(rows).to_csv('uma_out/uma_common_results.csv',index=False)
c=pd.DataFrame(rows); print(c[c.target_fpr==0.05].drop(columns=['target_fpr']).to_string(index=False))
