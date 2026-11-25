import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, pickle, numpy as np
import prefall_v3c as P
P.MODEL="hgb"; df=pd.read_pickle("real_v5/windows_cache.pkl"); COLS=P.L0+P.L1+P.L4
m=P.fit_rf(df,COLS)
pickle.dump(dict(model=m["rf"],cols=COLS,names=[P.NAMES[c] for c in COLS],thr_fpr=m["thr_fpr"],thr_default=m["thr"],note="HGB, 19 features, trained on all 38 SisFall subjects; thresholds = inner 3-fold subject-wise OOF ADL-window quantiles"),open("ltmm_pack/ltmm_model.pkl","wb"))
print("thr_fpr",m["thr_fpr"],"n windows",len(df),"ALL DONE")
