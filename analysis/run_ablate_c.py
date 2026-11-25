import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, json; sys.path.insert(0,'.')
import pandas as pd, numpy as np
import prefall_v3c as P
df=pd.read_pickle('real_v5/windows_cache.pkl')
rows=[]
for name,cols in [("C3 RF + second accelerometer",P.L0+P.L3),("C4 RF + activity context",P.L0+P.L4),("C5 RF all added groups",P.L0+P.L1+P.L3+P.L4)]:
    P.MODEL='rf'; r,_=P.run_cv(df,"rf",cols,10,0.1)
    row={"config":name,"AUC":P.ms(r.auc)}
    for k in (2,5,10):
        row[f"sens@{k}"]=P.ms(r[f"sens@{k}"],True); row[f"winFPR@{k}"]=P.ms(r[f"fpr_win@{k}"],True)
    rows.append(row); print(row,flush=True)
    pd.DataFrame(rows).to_csv('ablate_causal.csv',index=False)
print('ABLATE DONE',flush=True)
