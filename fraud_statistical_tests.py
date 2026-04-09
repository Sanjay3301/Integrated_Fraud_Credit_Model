import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading dataset...")

tx = pd.read_csv("Data/creditcard.csv")

X = tx.drop(columns=["Class"])
y = tx["Class"]

scaler = StandardScaler()
X[["Amount","Time"]] = scaler.fit_transform(X[["Amount","Time"]])

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

print("Loading trained models...")

logreg = joblib.load("thesis_results/models/logreg_best_model.joblib")
rf = joblib.load("thesis_results/models/rf_best_model.joblib")
xgb = joblib.load("thesis_results/models/xgb_best_model.joblib")

models = {
    "logreg":logreg,
    "rf":rf,
    "xgb":xgb
}

results=[]

def bootstrap_ci(y_true, y_prob, metric, n_boot=1000):

    scores=[]
    n=len(y_true)

    for i in range(n_boot):

        idx=np.random.randint(0,n,n)

        if len(np.unique(y_true.iloc[idx])) < 2:
            continue

        score=metric(y_true.iloc[idx],y_prob[idx])
        scores.append(score)

    lower=np.percentile(scores,2.5)
    upper=np.percentile(scores,97.5)

    return lower,upper,np.mean(scores),np.std(scores)

for name,model in models.items():

    print("Testing:",name)

    prob=model.predict_proba(X_test)[:,1]

    roc_ci=bootstrap_ci(y_test.reset_index(drop=True),prob,roc_auc_score)
    pr_ci=bootstrap_ci(y_test.reset_index(drop=True),prob,average_precision_score)

    results.append({
        "model":name,
        "roc_mean":roc_ci[2],
        "roc_std":roc_ci[3],
        "roc_ci_low":roc_ci[0],
        "roc_ci_high":roc_ci[1],
        "pr_mean":pr_ci[2],
        "pr_std":pr_ci[3],
        "pr_ci_low":pr_ci[0],
        "pr_ci_high":pr_ci[1]
    })

os.makedirs("thesis_results/tables",exist_ok=True)

df=pd.DataFrame(results)

df.to_csv("thesis_results/tables/statistical_validation.csv",index=False)

print("\nStatistical validation complete.")
print(df)