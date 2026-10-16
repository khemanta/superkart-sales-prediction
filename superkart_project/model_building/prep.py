"""Cleans the registered dataset, engineers features, splits it, and uploads the splits.

Run by the `data-prep` job of the GitHub Actions workflow.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "khemanta/superkart-sales-prediction"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/SuperKart.csv"
TARGET = "Product_Store_Sales_Total"
CURRENT_YEAR = 2026

# ---------------------------------------------------------------- load ------
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded from the Hub: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ------------------------------------------------- clean and engineer ------
# 1. The product identifier prefix encodes a category (FD food, NC non-consumable, DR drinks).
df["Product_Category"] = df["Product_Id"].str[:2]

# 2. Store age is a more useful feature than the raw establishment year.
df["Store_Age_Years"] = CURRENT_YEAR - df["Store_Establishment_Year"]

# 3. Fix the data-entry error: "reg" is the same category as "Regular".
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace("reg", "Regular")

# 4. Drop the columns that are now redundant or unusable.
df = df.drop(columns=["Product_Id", "Store_Establishment_Year"])

before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows; {len(df):,} remain")
print(f"Columns kept ({df.shape[1]}): {list(df.columns)}")

# --------------------------------------------------------------- split ------
X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {Xtrain.shape[0]:,} rows | mean sales {ytrain.mean():,.2f}")
print(f"Test : {Xtest.shape[0]:,} rows | mean sales {ytest.mean():,.2f}")

# ---------------------------------------------------- save and upload ------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path}")

print(f"Splits available at https://huggingface.co/datasets/{DATASET_REPO}")
