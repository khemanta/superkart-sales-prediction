"""Trains, tunes, tracks and registers the SuperKart sales model.

Run by the `model-training` job of the GitHub Actions workflow:
  1. reads the train/test splits from the Hugging Face dataset repo
  2. grid-searches an XGBoost pipeline, scoring on R2
  3. logs every parameter combination and the final metrics to MLflow
  4. uploads the best model to the Hugging Face model hub
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import mlflow

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("superkart-sales-prediction")

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "khemanta/superkart-sales-prediction"
MODEL_REPO = "khemanta/superkart-sales-model"
MODEL_FILE = "best_superkart_model_v1.joblib"
SEED = 42

# ----------------------------------------------------------------- data ----
base = f"hf://datasets/{DATASET_REPO}"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"{base}/ytest.csv").squeeze()
print(f"Loaded from the Hub - train {Xtrain.shape}, test {Xtest.shape}")

# ------------------------------------------------------------ pipeline -----
numeric_features = Xtrain.select_dtypes(include=["number"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include="object").columns.tolist()
print("numeric    :", numeric_features)
print("categorical:", categorical_features)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

model_pipeline = make_pipeline(preprocessor, xgb.XGBRegressor(random_state=SEED, n_jobs=-1))

param_grid = {
    "xgbregressor__n_estimators": [200, 300],
    "xgbregressor__max_depth": [3, 4, 5],
    "xgbregressor__learning_rate": [0.05, 0.1],
    "xgbregressor__subsample": [0.8],
    "xgbregressor__colsample_bytree": [0.8],
}

# ------------------------------------------------------- train and log -----
with mlflow.start_run(run_name="xgboost-gridsearch-ci"):
    grid_search = GridSearchCV(
        model_pipeline, param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="r2", n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_r2", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_r2", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_r2", grid_search.best_score_)
    best_model = grid_search.best_estimator_

    metrics = {}
    for label, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        y_pred = best_model.predict(X)
        metrics.update({
            f"{label}_r2": r2_score(y, y_pred),
            f"{label}_rmse": mean_squared_error(y, y_pred) ** 0.5,
            f"{label}_mae": mean_absolute_error(y, y_pred),
        })
    mlflow.log_metrics(metrics)

    print("Best parameters:", grid_search.best_params_)
    print(f"Best cross-validated R2: {grid_search.best_score_:.4f}")
    for key, value in metrics.items():
        print(f"   {key:12s} {value:,.4f}")

    joblib.dump(best_model, MODEL_FILE)
    mlflow.log_artifact(MODEL_FILE, artifact_path="model")
    print(f"Model saved as {MODEL_FILE}")

# ------------------------------------------------ register on the Hub ------
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{MODEL_REPO}' not found. Creating it...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Model registered at https://huggingface.co/{MODEL_REPO}")
