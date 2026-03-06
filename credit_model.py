# ================================
# CREDIT RISK MODEL – FINAL VERSION
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV


def run_credit_model():

    # ----------------------------
    # 1️⃣ Load Dataset
    # ----------------------------

    credit_data = pd.read_csv("Data/credit_raw.csv")

    if "Unnamed: 0" in credit_data.columns:
        credit_data = credit_data.drop("Unnamed: 0", axis=1)

    print("Dataset Shape:", credit_data.shape)
    print("Columns:", credit_data.columns)

    # ----------------------------
    # 2️⃣ Handle Missing Values
    # ----------------------------

    credit_data["Saving accounts"] = credit_data["Saving accounts"].fillna("none")
    credit_data["Checking account"] = credit_data["Checking account"].fillna("none")

    # ----------------------------
    # 3️⃣ Synthetic Risk Generation
    # ----------------------------

    np.random.seed(42)

    credit_amt_std = (
        credit_data["Credit amount"] - credit_data["Credit amount"].mean()
    ) / credit_data["Credit amount"].std()

    duration_std = (
        credit_data["Duration"] - credit_data["Duration"].mean()
    ) / credit_data["Duration"].std()

    age_std = (
        credit_data["Age"] - credit_data["Age"].mean()
    ) / credit_data["Age"].std()

    interaction = credit_amt_std * duration_std

    score = (
        0.9 * credit_amt_std
        + 0.8 * duration_std
        - 0.6 * age_std
        + 0.5 * interaction
        + 0.6 * (credit_data["Checking account"] == "none").astype(int)
        + 0.5 * (credit_data["Saving accounts"] == "none").astype(int)
    )

    prob = 1 / (1 + np.exp(-score))
    prob = prob + np.random.normal(0, 0.07, size=len(prob))
    prob = np.clip(prob, 0, 1)

    threshold = np.median(prob)
    credit_data["risk"] = (prob > threshold).astype(int)

    print("\nFull Dataset Risk Distribution:")
    print(credit_data["risk"].value_counts())

    # ----------------------------
    # 4️⃣ Train/Test Split
    # ----------------------------

    X = credit_data.drop("risk", axis=1)
    y = credit_data["risk"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTrain Distribution:")
    print(y_train.value_counts())

    print("\nTest Distribution:")
    print(y_test.value_counts())

    # ----------------------------
    # 5️⃣ Preprocessing
    # ----------------------------

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    # ================================
    # 6️⃣ Random Forest – Hyperparameter Tuning
    # ================================

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    param_dist = {
        "classifier__n_estimators": [200, 300, 400],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }

    rf_random = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=15,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )

    rf_random.fit(X_train, y_train)

    print("\nBest RF Params:", rf_random.best_params_)

    best_rf = rf_random.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

    print("\n===== Tuned Random Forest Results =====")
    print(classification_report(y_test, y_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

    # ================================
    # 7️⃣ Logistic Regression Baseline
    # ================================

    log_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, solver="liblinear"))
    ])

    log_pipeline.fit(X_train, y_train)

    y_pred_log = log_pipeline.predict(X_test)
    y_proba_log = log_pipeline.predict_proba(X_test)[:, 1]

    print("\n===== Logistic Regression Results =====")
    print(classification_report(y_test, y_pred_log))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))

    # ================================
    # 8️⃣ Calibrated Logistic Regression
    # ================================

    calibrated_log = CalibratedClassifierCV(
        log_pipeline,
        method="sigmoid",
        cv=5
    )

    calibrated_log.fit(X_train, y_train)

    y_proba_cal = calibrated_log.predict_proba(X_test)[:, 1]

    print("\n===== Calibrated Logistic Regression =====")
    print("Calibrated ROC-AUC:", roc_auc_score(y_test, y_proba_cal))

    # ----------------------------
    # 9️⃣ Feature Importance
    # ----------------------------

    os.makedirs("results", exist_ok=True)

    cat_feature_names = best_rf.named_steps["preprocessor"] \
        .named_transformers_["cat"] \
        .named_steps["encoder"] \
        .get_feature_names_out(cat_cols)

    feature_names = list(num_cols) + list(cat_feature_names)

    importances = best_rf.named_steps["classifier"].feature_importances_

    importance_df = pd.Series(importances, index=feature_names)

    importance_df.nlargest(10).sort_values().plot(kind="barh")

    plt.title("Top Credit Risk Features (Tuned RF)")
    plt.tight_layout()
    plt.savefig("results/credit_feature_importance.png", dpi=300)
    plt.show()

    # ----------------------------
    # 🔟 Integration Output
    # ----------------------------

    credit_output = X_test.copy()

    credit_output["actual_risk"] = y_test.values
    credit_output["credit_probability"] = y_proba_cal

    credit_output["credit_score"] = -50 * np.log(
        1 - credit_output["credit_probability"] + 1e-6
    )

    def risk_band(score):

        if score < 30:
            return "Low"

        elif score < 60:
            return "Medium"

        else:
            return "High"

    credit_output["risk_band"] = credit_output["credit_score"].apply(risk_band)

    credit_output.to_csv(
        "results/credit_integration_output.csv",
        index=False
    )

    print("\nIntegration-ready credit output saved.")

    return credit_output


if __name__ == "__main__":
    run_credit_model()