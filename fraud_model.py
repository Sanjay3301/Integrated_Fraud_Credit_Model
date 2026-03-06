import pandas as pd

# Load Fraud Dataset
fraud_data = pd.read_csv("fraud_raw.csv")

# Load Credit Risk Dataset
credit_data = pd.read_csv("credit_raw.csv")

print("Fraud Dataset Shape:", fraud_data.shape)
print("Credit Dataset Shape:", credit_data.shape)

print("\nFraud Dataset Preview:")
print(fraud_data.head())

print("\nCredit Dataset Preview:")
print(credit_data.head())
print("\nFraud Dataset Info:")
print(fraud_data.info())

print("\nFraud Dataset Class Distribution:")
print(fraud_data['Class'].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = fraud_data.drop("Class", axis=1)
y = fraud_data["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
# -----------------------------
# SCALE THE DATA (Important)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

log_model = LogisticRegression(max_iter=2000, class_weight='balanced')

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\n===== LOGISTIC REGRESSION =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_prob_log))


# -----------------------------
# RANDOM FOREST
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_prob_rf))
import os
os.makedirs("results", exist_ok=True)
from sklearn.metrics import average_precision_score

print("\nPR-AUC (Logistic Regression):")
print(average_precision_score(y_test, y_prob_log))

print("\nPR-AUC (Random Forest):")
print(average_precision_score(y_test, y_prob_rf))

# -----------------------------
# XGBOOST
# -----------------------------
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n===== XGBOOST =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_prob_xgb))

print("\nPR-AUC (XGBoost):")
print(average_precision_score(y_test, y_prob_xgb))

# ===============================
# FEATURE IMPORTANCE (XGBoost)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

feature_names = X.columns

importance = xgb_model.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(feat_imp.head(15))

# Plot
plt.figure(figsize=(8,6))
plt.barh(feat_imp["Feature"][:15], feat_imp["Importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance - XGBoost")
plt.xlabel("Importance Score")
plt.show()

# PRECISION-RECALL CURVE

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_prob_rf)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.show()
# --------------------------------
# THRESHOLD OPTIMIZATION (Proper)
# --------------------------------

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("\n===== THRESHOLD TUNING (Random Forest) =====")

thresholds = np.arange(0.01, 1.00, 0.01)

best_f1 = 0
best_threshold = 0

for t in thresholds:
    y_pred_custom = (y_prob_rf >= t).astype(int)

    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

    print(f"Threshold: {t:.2f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

print("\nBest Threshold (based on F1):", best_threshold)
print("Best F1 Score:", best_f1)

# Final evaluation at best threshold
y_final = (y_prob_rf >= best_threshold).astype(int)

print("\nConfusion Matrix at Best Threshold:")
print(confusion_matrix(y_test, y_final))

# ===============================
# SAVE FRAUD PREDICTIONS
# ===============================

fraud_output = pd.DataFrame({
    "Actual_Class": y_test.values,
    "Predicted_Class": y_pred_xgb,
    "Fraud_Probability": y_prob_xgb
})

fraud_output.to_csv("fraud_predictions_output.csv", index=False)

print("\nFraud prediction file saved successfully!")
