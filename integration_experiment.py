import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading fraud dataset...")

tx = pd.read_csv("Data/creditcard.csv")

X_tx = tx.drop(columns=["Class"])
y_tx = tx["Class"]

scaler = StandardScaler()
X_tx[["Amount","Time"]] = scaler.fit_transform(X_tx[["Amount","Time"]])

X_train_tx,X_test_tx,y_train_tx,y_test_tx = train_test_split(
    X_tx,y_tx,test_size=0.2,stratify=y_tx,random_state=42
)

print("Loading credit dataset...")

cr = pd.read_csv("Data/german_credit.csv")

X_cr = cr.iloc[:,:-1]
y_cr = cr.iloc[:,-1]

X_cr = StandardScaler().fit_transform(X_cr)

X_train_cr,X_test_cr,y_train_cr,y_test_cr = train_test_split(
    X_cr,y_cr,test_size=0.2,stratify=y_cr,random_state=42
)

print("Loading trained models...")

fraud_model = joblib.load("thesis_results/models/xgb_best_model.joblib")
credit_model = joblib.load("thesis_results/models/credit_rf.joblib")

Pf = fraud_model.predict_proba(X_test_tx)[:,1]
Pd = credit_model.predict_proba(X_test_cr)[:,1]

# Match sizes
n = min(len(Pf),len(Pd))

Pf = Pf[:n]
Pd = Pd[:n]

alpha = 0.6
beta = 0.4

URS = alpha*Pf + beta*Pd

df = pd.DataFrame({
    "fraud_probability":Pf,
    "credit_probability":Pd,
    "URS":URS
})

def decision(score):

    if score < 0.3:
        return "Approve"
    elif score < 0.6:
        return "Manual Review"
    else:
        return "Reject"

df["decision"] = df["URS"].apply(decision)

os.makedirs("thesis_results/tables",exist_ok=True)

df.to_csv("thesis_results/tables/URS_results.csv",index=False)

print("\nIntegrated risk system complete.")

print(df["decision"].value_counts())