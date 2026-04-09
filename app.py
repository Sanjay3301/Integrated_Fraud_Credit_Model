import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Risk Engine", layout="wide")
st.title("💳 Integrated Fraud + Credit Risk System")

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "demo_app", "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "demo_app", "sample_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "demo_app", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    fraud_model = joblib.load(os.path.join(MODEL_DIR, "xgb_best_model.joblib"))
    credit_model = joblib.load(os.path.join(MODEL_DIR, "credit_rf.joblib"))
    return fraud_model, credit_model

fraud_model, credit_model = load_models()

# =========================
# Expected schema
# =========================
FRAUD_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Options")

mode = st.sidebar.radio("Choose Mode", ["Demo", "Upload"])

st.sidebar.subheader("🎛 Decision Thresholds")
approve_th = st.sidebar.slider("Approve Threshold", 0.0, 1.0, 0.3)
review_th = st.sidebar.slider("Review Threshold", 0.0, 1.0, 0.6)

# =========================
# Helpers
# =========================
def load_demo_data():
    fraud = pd.read_csv(os.path.join(SAMPLE_DIR, "sample.csv"))
    credit = pd.read_csv(os.path.join(SAMPLE_DIR, "credit_sample.csv"))
    return fraud, credit

def prepare_fraud(df):
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    missing = [c for c in FRAUD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing fraud columns: {missing}")

    return df[FRAUD_COLUMNS]

def prepare_credit(df):
    if df.shape[1] == 25:
        df = df.iloc[:, :-1]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    if df.shape[1] != 24:
        raise ValueError(f"Credit data must have 24 features, got {df.shape[1]}")

    return df

def decision_logic(x):
    if x < approve_th:
        return "Approve"
    elif x < review_th:
        return "Manual Review"
    else:
        return "Reject"

def explain_row(row):
    reasons = []
    if row["fraud_probability"] > 0.7:
        reasons.append("High fraud risk")
    if row["credit_probability"] > 0.7:
        reasons.append("High credit risk")
    if row["URS"] > review_th:
        reasons.append("Overall high risk")
    if not reasons:
        reasons.append("Low risk profile")
    return ", ".join(reasons)

# =========================
# Load Data
# =========================
if mode == "Demo":
    try:
        fraud_df, credit_df = load_demo_data()
        st.success("Using demo datasets")
    except:
        st.error("Demo files missing")
        st.stop()

else:
    st.subheader("Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        fraud_file = st.file_uploader("Upload Fraud CSV", type=["csv"])

    with col2:
        credit_file = st.file_uploader("Upload Credit CSV", type=["csv"])

    if fraud_file and credit_file:
        fraud_df = pd.read_csv(fraud_file)
        credit_df = pd.read_csv(credit_file)
    else:
        st.warning("Upload both files")
        st.stop()

# =========================
# Preview
# =========================
st.subheader("📊 Fraud Data")
st.dataframe(fraud_df.head())

st.subheader("📊 Credit Data")
st.dataframe(credit_df.head())

# =========================
# Run Analysis
# =========================
if st.button("🚀 Run Risk Analysis"):

    with st.spinner("Running models..."):

        # FRAUD
        Xf = prepare_fraud(fraud_df)
        Pf = fraud_model.predict_proba(Xf)[:, 1]

        # CREDIT
        Xc = prepare_credit(credit_df)
        Pd = credit_model.predict_proba(Xc)[:, 1]

        # ALIGN
        n = min(len(Pf), len(Pd))
        Pf, Pd = Pf[:n], Pd[:n]

        # URS
        URS = 0.6 * Pf + 0.4 * Pd
        decisions = [decision_logic(x) for x in URS]

        result = pd.DataFrame({
            "fraud_probability": Pf,
            "credit_probability": Pd,
            "URS": URS,
            "decision": decisions
        })

        # Explainability
        result["explanation"] = result.apply(explain_row, axis=1)

    # =========================
    # KEY METRICS
    # =========================
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Fraud Risk", f"{Pf.mean():.4f}")
    col2.metric("Avg Credit Risk", f"{Pd.mean():.4f}")
    col3.metric("Avg URS", f"{URS.mean():.4f}")
    col4.metric("Total Records", len(result))

    # =========================
    # Decision Summary
    # =========================
    st.subheader("📌 Decision Summary")

    decision_counts = result["decision"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("Approve", decision_counts.get("Approve", 0))
    col2.metric("Manual Review", decision_counts.get("Manual Review", 0))
    col3.metric("Reject", decision_counts.get("Reject", 0))

    # =========================
    # Risk Segmentation
    # =========================
    st.subheader("📊 Risk Segmentation")

    bins = [0, approve_th, review_th, 1]
    labels = ["Low Risk", "Medium Risk", "High Risk"]

    result["risk_segment"] = pd.cut(result["URS"], bins=bins, labels=labels)

    st.bar_chart(result["risk_segment"].value_counts())

    # =========================
    # Results Table
    # =========================
    st.subheader("📋 Results with Explanation")
    st.dataframe(result)

    # =========================
    # Charts
    # =========================
    st.subheader("📈 Risk Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Fraud Probability")
        st.bar_chart(result["fraud_probability"])

    with col2:
        st.write("URS Score")
        st.bar_chart(result["URS"])

    # =========================
    # Top Risk Cases
    # =========================
    st.subheader("🚨 Top High Risk Cases")
    st.dataframe(result.sort_values("URS", ascending=False).head(10))

    # =========================
    # Download High Risk
    # =========================
    st.subheader("📥 Download High Risk Cases")

    high_risk = result[result["decision"] == "Reject"]
    csv_high = high_risk.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download High Risk Cases",
        csv_high,
        "high_risk_cases.csv",
        "text/csv"
    )

    # =========================
    # Save + Download All
    # =========================
    output_path = os.path.join(OUTPUT_DIR, "results.csv")
    result.to_csv(output_path, index=False)

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download All Results",
        csv,
        "risk_results.csv",
        "text/csv"
    )

    st.success(f"Saved to: {output_path}")