import pandas as pd

# Load numeric German dataset (handle multiple spaces)
df = pd.read_csv("Data/german.data-numeric", sep=r"\s+", engine="python", header=None)

# Save as CSV
df.to_csv("Data/german_credit.csv", index=False)

print("German dataset converted successfully!")
print("Shape:", df.shape)