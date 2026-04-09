import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

RSEED = 42
np.random.seed(RSEED)

os.makedirs("thesis_results/models", exist_ok=True)
os.makedirs("thesis_results/tables", exist_ok=True)

print("Loading fraud dataset...")

tx = pd.read_csv("Data/creditcard.csv")

X = tx.drop(columns=["Class"])
y = tx["Class"]

scaler = StandardScaler()
X[["Amount","Time"]] = scaler.fit_transform(X[["Amount","Time"]])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RSEED
)

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RSEED)

models = {

"logreg":(
Pipeline([
("clf",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=RSEED))
]),
{"clf__C":[0.01,0.1,1]}
),

"rf":(
Pipeline([
("clf",RandomForestClassifier(random_state=RSEED))
]),
{
"clf__n_estimators":[100,300],
"clf__max_depth":[6,10,None]
}
),

"xgb":(
Pipeline([
("clf",xgb.XGBClassifier(eval_metric="logloss",random_state=RSEED))
]),
{
"clf__n_estimators":[100,300],
"clf__max_depth":[4,6],
"clf__scale_pos_weight":[1,10,50]
}
)

}

scoring = {
"roc_auc":"roc_auc",
"average_precision":"average_precision"
}

results=[]

for name,(pipe,grid) in models.items():

    print(f"Running {name} model...")

    gscv=GridSearchCV(pipe,grid,cv=inner,scoring="average_precision",n_jobs=-1)

    cvres=cross_validate(
        gscv,
        X_train,
        y_train,
        cv=outer,
        scoring=scoring,
        return_estimator=True,
        n_jobs=-1
    )

    roc=cvres["test_roc_auc"]
    pr=cvres["test_average_precision"]

    results.append({
        "model":name,
        "roc_auc_mean":roc.mean(),
        "roc_auc_std":roc.std(),
        "pr_auc_mean":pr.mean(),
        "pr_auc_std":pr.std()
    })

    best=cvres["estimator"][0].best_estimator_

    joblib.dump(best,f"thesis_results/models/{name}_best_model.joblib")

summary=pd.DataFrame(results)

summary.to_csv("thesis_results/tables/fraud_model_comparison_cv.csv",index=False)

print("\nModel comparison saved!")
print(summary)