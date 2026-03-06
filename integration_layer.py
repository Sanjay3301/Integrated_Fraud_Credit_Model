# ==========================================
# INTEGRATED FRAUD + CREDIT RISK SYSTEM
# ==========================================

import pandas as pd
import numpy as np

from fraud_model import run_fraud_model
from credit_model import run_credit_model


def run_integration():

    # ----------------------------------
    # 1️⃣ Run Fraud Model
    # ----------------------------------

    print("\nRunning Fraud Model...")

    try:
        fraud_output = run_fraud_model()

    except FileNotFoundError:

        print("fraud_raw.csv not found — using simulated fraud probabilities")

        # We simulate fraud probabilities to keep pipeline functional
        np.random.seed(42)

        fraud_output = pd.DataFrame()

        # temporary length (will align with credit output later)
        fraud_output["fraud_probability"] = np.random.beta(
            a=0.5,
            b=10,
            size=200
        )

    # ----------------------------------
    # 2️⃣ Run Credit Model
    # ----------------------------------

    print("\nRunning Credit Model...")

    credit_output = run_credit_model()

    # ----------------------------------
    # 3️⃣ Align Dataset Lengths
    # ----------------------------------

    min_len = min(len(fraud_output), len(credit_output))

    fraud_output = fraud_output.iloc[:min_len].reset_index(drop=True)
    credit_output = credit_output.iloc[:min_len].reset_index(drop=True)

    # ----------------------------------
    # 4️⃣ Merge Model Outputs
    # ----------------------------------

    integration_df = pd.concat(
        [
            fraud_output["fraud_probability"],
            credit_output[["credit_probability", "credit_score", "risk_band"]],
        ],
        axis=1,
    )

    # ----------------------------------
    # 5️⃣ Combined Risk Score
    # ----------------------------------

    integration_df["combined_risk"] = (
        0.6 * integration_df["fraud_probability"]
        + 0.4 * integration_df["credit_probability"]
    )

    # ----------------------------------
    # 6️⃣ Decision Engine
    # ----------------------------------

    def final_decision(row):

        if row["fraud_probability"] > 0.90:
            return "Reject"

        if row["risk_band"] == "Low":
            return "Approve"

        elif row["risk_band"] == "Medium":
            return "Manual Review"

        else:
            return "Reject"

    integration_df["final_decision"] = integration_df.apply(final_decision, axis=1)

    # ----------------------------------
    # 7️⃣ Diagnostics
    # ----------------------------------

    print("\nFraud Probability Stats:")
    print(integration_df["fraud_probability"].describe())

    print("\nCredit Risk Band Distribution:")
    print(integration_df["risk_band"].value_counts())

    print("\nFinal Decision Distribution:")
    print(integration_df["final_decision"].value_counts())

    # ----------------------------------
    # 8️⃣ Save Output
    # ----------------------------------

    integration_df.to_csv("results/integrated_decisions.csv", index=False)

    print("\nIntegration Complete.")

    return integration_df


# ----------------------------------
# Run if executed directly
# ----------------------------------

if __name__ == "__main__":
    run_integration()