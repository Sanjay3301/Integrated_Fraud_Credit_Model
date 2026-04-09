import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

print("Loading fraud dataset...")

tx = pd.read_csv("Data/creditcard.csv")

X = tx.drop(columns=["Class"])
y = tx["Class"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[["Amount","Time"]] = scaler.fit_transform(X[["Amount","Time"]])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

models=["logreg","rf","xgb"]

os.makedirs("thesis_results/figures",exist_ok=True)

plt.figure()

for name in models:

    model=joblib.load(f"thesis_results/models/{name}_best_model.joblib")

    prob=model.predict_proba(X_test)[:,1]
    pred=model.predict(X_test)

    fpr,tpr,_=roc_curve(y_test,prob)
    roc_auc=auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.3f})")

    cm=confusion_matrix(y_test,pred)

    plt.figure()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title(f"Confusion Matrix ({name})")
    plt.savefig(f"thesis_results/figures/confusion_{name}.png")
    plt.close()

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("thesis_results/figures/roc_curve.png")
plt.close()

print("Evaluation figures saved.")