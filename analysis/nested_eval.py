import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
import prefall_v3c as P
D=pd.read_pickle("analysis_causal/oof_nested.pkl"); H=pd.read_pickle("analysis_causal/oof_C2_HGB_all.pkl")
def summ(d,tag):
    y=d.pos.values; s=d.score.values; auc=roc_auc_score(y,s); fa=d.groupby("fold").apply(lambda g:roc_auc_score(g.pos,g.score))
    thr=d["thr5"].values; adl=d.adl.values==1
    m=P.op_metrics(s,d.trial.values,y==1,adl,d.dt.values,thr)
    us=d.subject.unique(); ix={u:np.where(d.subject.values==u)[0] for u in us}; rng=np.random.default_rng(5); b=[]
    for _ in range(500):
        ii=np.concatenate([ix[u] for u in rng.choice(us,len(us))]); b.append(roc_auc_score(y[ii],s[ii]))
    return dict(model=tag,AUC_pooled=round(auc,3),AUC_lo=round(np.percentile(b,2.5),3),AUC_hi=round(np.percentile(b,97.5),3),AUC_fold_mean=round(fa.mean(),3),AUC_fold_sd=round(fa.std(),3),
        trial_sens5=round(m["sens"],3),window_fpr5=round(m["fpr_win"],3),mean_warning_s=round(m["warn"],2),n_fall=m["n_fall"])
rows=[summ(H,"non-nested HGB (fixed config)"),summ(D,"nested (inner-selected)")]
pd.DataFrame(rows).to_csv("analysis_causal/item15_nested_vs_fixed.csv",index=False); print(pd.DataFrame(rows).to_string())
print(pd.Series([s["chosen"] for s in json.load(open("analysis_causal/item15_nested_selection.json"))]).value_counts())
