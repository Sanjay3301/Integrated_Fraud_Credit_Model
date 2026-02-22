import pandas as pd

# Load credit dataset
credit_data = pd.read_csv("Data/credit_raw.csv")

print("Shape:", credit_data.shape)
print("\nPreview:")
print(credit_data.head())

print("\nInfo:")
print(credit_data.info())

print("\nMissing values:")
print(credit_data.isnull().sum())

credit_data = credit_data.drop("Unnamed: 0", axis=1)
credit_data["Saving accounts"] = credit_data["Saving accounts"].fillna("none")
credit_data["Checking account"] = credit_data["Checking account"].fillna("none")
credit_data = pd.get_dummies(credit_data, drop_first=True)
print("\nAfter encoding shape:", credit_data.shape)
print(credit_data.head())


# Create risk label (simple proxy using credit amount & duration)

credit_data["risk"] = (
    (credit_data["Credit amount"] > credit_data["Credit amount"].median()) &
    (credit_data["Duration"] > credit_data["Duration"].median())
).astype(int)

print("\nRisk distribution:")
print(credit_data["risk"].value_counts())

# Split data into train/test sets
from sklearn.model_selection import train_test_split

X = credit_data.drop("risk", axis=1)
y = credit_data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Baseline Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC:", roc_auc_score(y_test, y_pred))

import matplotlib.pyplot as plt
import pandas as pd

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.nlargest(10).plot(kind="barh")
plt.title("Top Credit Risk Features")
plt.show()
plt.tight_layout()
plt.savefig("results/credit_feature_importance.png", dpi=300)
plt.show()