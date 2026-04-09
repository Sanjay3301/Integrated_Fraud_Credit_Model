import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

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

print("Loading trained XGBoost model...")
pipeline = joblib.load("thesis_results/models/xgb_best_model.joblib")

model = pipeline.named_steps["clf"]

print("Computing SHAP values...")

explainer = shap.TreeExplainer(model)

sample = X_test.sample(2000, random_state=42)

shap_values = explainer.shap_values(sample)

os.makedirs("thesis_results/figures",exist_ok=True)
os.makedirs("thesis_results/tables",exist_ok=True)

print("Computing SHAP values...")

explainer = shap.TreeExplainer(model)

sample = X_test.sample(2000, random_state=42)

shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.savefig("thesis_results/figures/shap_summary.png", bbox_inches="tight")
plt.close()

importance = pd.DataFrame({
    "feature":sample.columns,
    "importance":np.abs(shap_values).mean(axis=0)
}).sort_values("importance",ascending=False)

importance.to_csv("thesis_results/tables/feature_importance.csv",index=False)

print("SHAP analysis complete.")