import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print("Loading dataset...")

tx = pd.read_csv("Data/creditcard.csv")

X = tx.drop(columns=["Class"])
y = tx["Class"]

scaler = StandardScaler()
X[["Amount","Time"]] = scaler.fit_transform(X[["Amount","Time"]])

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

print("Loading model...")

model = joblib.load("thesis_results/models/xgb_best_model.joblib")

prob_original = model.predict_proba(X_test)[:,1]

auc_original = roc_auc_score(y_test,prob_original)

print("Original AUC:",auc_original)

# add noise
noise = np.random.normal(0,0.05,X_test.shape)

X_noisy = X_test + noise

prob_noisy = model.predict_proba(X_noisy)[:,1]

auc_noisy = roc_auc_score(y_test,prob_noisy)

print("Noisy AUC:",auc_noisy)