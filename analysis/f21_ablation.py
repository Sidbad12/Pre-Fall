import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, json
import prefall_v3c as P
df=pd.read_pickle("real_v5/windows_cache.pkl"); P.MODEL="hgb"
ALL=list(range(22)); res=[]
for name,cols in (("all 22",ALL),("without f21 (previous-activity intensity)",[c for c in ALL if c!=20]),
                  ("without all context f19-f22",[c for c in ALL if c not in (18,19,20,21)]),
                  ("without f19-f22 and second accelerometer",[c for c in ALL if c not in (15,16,17,18,19,20,21)])):
    r,_=P.run_cv(df,"rf",cols,10,None)
    res.append(dict(config=name,n_features=len(cols),AUC=round(r.auc.mean(),3),AUC_sd=round(r.auc.std(),3),sens5=round(r["sens@5"].mean(),3),fpr5=round(r["fpr_win@5"].mean(),3)))
    print(res[-1],flush=True); pd.DataFrame(res).to_csv("analysis_causal/item13_f21_ablation.csv",index=False)
