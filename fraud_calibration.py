import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
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

print("Loading XGBoost model...")
pipeline = joblib.load("thesis_results/models/xgb_best_model.joblib")
model = pipeline.named_steps["clf"]

probs = model.predict_proba(X_test)[:,1]

brier = brier_score_loss(y_test,probs)

print("Brier Score:", brier)

os.makedirs("thesis_results/figures",exist_ok=True)
os.makedirs("thesis_results/tables",exist_ok=True)

prob_true,prob_pred = calibration_curve(y_test,probs,n_bins=10)

plt.figure()

plt.plot(prob_pred,prob_true,marker="o",label="XGBoost")
plt.plot([0,1],[0,1],"--",label="Perfect calibration")

plt.xlabel("Predicted probability")
plt.ylabel("True probability")

plt.title("Calibration Curve")

plt.legend()

plt.savefig("thesis_results/figures/calibration_curve.png")

pd.DataFrame({
    "model":["xgboost"],
    "brier_score":[brier]
}).to_csv("thesis_results/tables/brier_scores.csv",index=False)

print("Calibration analysis complete.")