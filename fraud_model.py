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
    n_estimators=50,
    random_state=42,
    class_weight='balanced',
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