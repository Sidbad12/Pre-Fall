import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, sys
import prefall_v3c as P
root=os.environ.get("SISFALL_ROOT","SisFall_dataset")
tab=pd.read_pickle("real_v5/windows_cache.pkl")[["trial","subject","act","is_fall","end","impact","dt","pos","train_neg","adl"]]
oof=pd.read_pickle("analysis_causal/oof_C2_HGB_all.pkl"); fmap=oof.groupby("subject").fold.first().to_dict()
tab["fold"]=tab.subject.map(fmap); print(tab.shape, tab.fold.isna().mean())
items=P.list_sisfall(root); X=[]; keep=[]
need={t:g for t,g in tab.groupby("trial")}
for i,it in need.items():
    acc,gyr,_=P.load_sisfall_file(items[i]["path"])
    a=P.causal_lp(acc); g=P.causal_lp(gyr); S=np.concatenate([a,g/500.0],1)   # gyro scaled to ~[-1,1]
    for e,idx in zip(it.end.values,it.index):
        e_=int(round(e*P.FS)); w=S[e_-P.WIN:e_][::4]                          # 200->50 Hz, 100 samples
        X.append(w.astype(np.float16)); keep.append(idx)
X=np.stack(X); m=tab.loc[keep].reset_index(drop=True)
np.savez_compressed("kaggle_pack/sisfall_raw_windows.npz",X=X); m.to_csv("kaggle_pack/meta.csv",index=False); print(X.shape, m.shape)
