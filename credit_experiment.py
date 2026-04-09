import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("Loading German credit dataset...")

df = pd.read_csv("Data/german_credit.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

outer = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
inner = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

models = {

"logreg":(
Pipeline([
("clf",LogisticRegression(max_iter=1000))
]),
{"clf__C":[0.01,0.1,1]}
),

"rf":(
Pipeline([
("clf",RandomForestClassifier(random_state=42))
]),
{
"clf__n_estimators":[100,300],
"clf__max_depth":[5,10,None]
}
)

}

results=[]

for name,(pipe,grid) in models.items():

    print("Running:",name)

    gscv=GridSearchCV(pipe,grid,cv=inner,n_jobs=-1)

    cvres=cross_validate(
        gscv,
        X_train,
        y_train,
        cv=outer,
        scoring=["accuracy","roc_auc"],
        return_estimator=True,
        n_jobs=-1
    )

    results.append({

        "model":name,
        "accuracy_mean":cvres["test_accuracy"].mean(),
        "accuracy_std":cvres["test_accuracy"].std(),
        "roc_auc_mean":cvres["test_roc_auc"].mean(),
        "roc_auc_std":cvres["test_roc_auc"].std()

    })

    best=cvres["estimator"][0].best_estimator_

    joblib.dump(best,f"thesis_results/models/credit_{name}.joblib")

os.makedirs("thesis_results/tables",exist_ok=True)

summary=pd.DataFrame(results)

summary.to_csv("thesis_results/tables/credit_model_comparison.csv",index=False)

print("\nCredit model experiment complete.")
print(summary)