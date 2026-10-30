"""Leave-house-out factorial: LR, DT and XGBoost x two feature sets x four imbalance strategies x five appliances
(120 configurations), on the fixed house partition used for the 35-configuration validation.
Hyperparameters fixed to the modal pooled nested-CV values. Resumable: finished folds are skipped.
USAGE (from nilm/src): python3 run_lho_factorial.py --plan ../results_lho_factorial/plan_factorial.json
       --partition ../results_lho_factorial/house_partition.csv --features-dir ../data/features --out ../results_lho_factorial"""
import argparse, os, json, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from model_pipeline import _prepare_estimator, RANDOM_STATE
ap=argparse.ArgumentParser()
for k in ["--plan","--partition","--features-dir","--out"]: ap.add_argument(k,required=True)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); fp=os.path.join(a.out,"fold_results.csv")
plan=json.load(open(a.plan)); part=pd.read_csv(a.partition)
ORD=["LogisticRegression","DecisionTree","XGBoost"]; IMB=["none","class_weight","SMOTE","ADASYN"]
plan=sorted(plan,key=lambda p:(p["appliance"],p["arm"],ORD.index(p["model"]),IMB.index(p["imbalance"])))
rows=pd.read_csv(fp).to_dict("records") if os.path.exists(fp) else []
done={(r["appliance"],r["arm"],r["model"],r["imbalance"],r["fold"]) for r in rows}
cur=None
for p in plan:
    app,arm,model,imb=p["appliance"],p["arm"],p["model"],p["imbalance"]
    if all((app,arm,model,imb,k) in done for k in range(1,6)): continue
    if cur!=(app,arm):
        df=pd.read_csv(os.path.join(a.features_dir,f"{app}_arm{arm}.csv")); cur=(app,arm)
        X=df.drop(columns=["label","house_id"]).values; y=df["label"].values.astype(int); g=df["house_id"].values; del df
    for k in range(1,6):
        if (app,arm,model,imb,k) in done: continue
        r=part[(part.appliance==app)&(part.fold==k)].iloc[0]; hs=[int(h) for h in str(r.test_houses).split()]
        te=np.where(np.isin(g,hs))[0]; tr=np.where(~np.isin(g,hs))[0]
        assert len(te)==int(r.n_test), (app,k,len(te),r.n_test)
        t0=time.time(); est=_prepare_estimator(model,imb,y[tr]); est.set_params(**p["params"])
        if model=="XGBoost": est.set_params(n_jobs=4)
        if imb in ("SMOTE","ADASYN"):
            s=SMOTE(random_state=RANDOM_STATE) if imb=="SMOTE" else ADASYN(random_state=RANDOM_STATE)
            est=ImbPipeline([("sampler",s),("model",est)])
        est.fit(X[tr],y[tr]); pr=est.predict(X[te]); yt=y[te]
        rows.append(dict(appliance=app,arm=arm,model=model,imbalance=imb,fold=k,test_houses=r.test_houses,n_test=len(te),
            n_test_on=int(yt.sum()),f1=f1_score(yt,pr,zero_division=0),precision=precision_score(yt,pr,zero_division=0),
            recall=recall_score(yt,pr,zero_division=0),accuracy=accuracy_score(yt,pr),tn=int(((yt==0)&(pr==0)).sum()),
            fp=int(((yt==0)&(pr==1)).sum()),fn=int(((yt==1)&(pr==0)).sum()),tp=int(((yt==1)&(pr==1)).sum()),
            elapsed_seconds=round(time.time()-t0,1)))
        pd.DataFrame(rows).to_csv(fp,index=False)
        print(f"{len(rows)}/600 {app} {arm} {model} {imb} fold {k} F1={rows[-1]['f1']:.4f} {rows[-1]['elapsed_seconds']}s",flush=True)
print("ALL DONE",flush=True)
