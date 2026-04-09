import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Loading URS results...")

df = pd.read_csv("thesis_results/tables/URS_results.csv")

# Cost assumptions
C_FN_FRAUD = 2000
C_FP_INVESTIGATION = 20
C_DEFAULT = 5000

expected_costs = []

for i,row in df.iterrows():

    fraud = row["fraud_probability"]
    credit = row["credit_probability"]
    decision = row["decision"]

    cost = 0

    if decision == "Manual Review":
        cost += C_FP_INVESTIGATION

    if decision == "Approve" and fraud > 0.5:
        cost += C_FN_FRAUD

    if decision == "Approve" and credit > 0.5:
        cost += C_DEFAULT

    expected_costs.append(cost)

df["cost"] = expected_costs

total_cost = df["cost"].sum()

print("\nTotal Expected Cost:", total_cost)

summary = df.groupby("decision")["cost"].sum()

print("\nCost by decision:")
print(summary)

os.makedirs("thesis_results/figures",exist_ok=True)

summary.plot(kind="bar")

plt.title("Expected Cost by Decision Type")
plt.ylabel("Cost")
plt.xlabel("Decision")

plt.savefig("thesis_results/figures/expected_costs.png")

print("\nCost analysis complete.")