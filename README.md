# SuperKart — Product Store Sales Prediction

An end-to-end MLOps pipeline that forecasts the revenue a given product will generate in a given
SuperKart store, so procurement and shelf allocation can be planned before stock is ordered.

Every stage — data registration, preparation, training, registration and deployment — runs as a
script, and the same four scripts run unattended on every push through GitHub Actions.

## Live artefacts

| What | Where |
|---|---|
| Dataset (raw + the four splits) | https://huggingface.co/datasets/khemanta/superkart-sales-prediction |
| Registered model | https://huggingface.co/khemanta/superkart-sales-model |
| Front end (Gradio Space) | https://huggingface.co/spaces/khemanta/superkart-sales-frontend |
| CI/CD workflow | [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) |

## Results

Tuned XGBoost regressor inside a scikit-learn pipeline (`max_depth` 5, 300 trees, learning rate
0.05, subsample and colsample 0.8), selected by 5-fold grid search over 12 combinations.

| Split | R² | RMSE | MAE |
|---|---|---|---|
| Train | 0.957 | 221.2 | 90.9 |
| Test | **0.926** | **290.6** | **120.8** |

Mean sales in the test set are ~3,461, so a typical error of ~291 is about 8%.

## Pipeline

```
superkart_project/
├── data/SuperKart.csv            raw data, 8,763 rows
├── model_building/
│   ├── data_register.py          creates the dataset repo and uploads the raw CSV
│   ├── prep.py                   cleaning, feature engineering, 80/20 split, uploads 4 CSVs
│   └── train.py                  grid search + MLflow tracking, uploads the .joblib model
├── deployment/                   Streamlit + Dockerfile front end (Docker Spaces need HF PRO)
├── deployment_gradio/            Gradio front end actually deployed, same model and inputs
└── hosting/
    ├── hosting.py                pushes the Docker Space  (returns 402 on a free account)
    └── hosting_gradio.py         pushes the Gradio Space  (the one in use)
```

### Feature engineering

- `Product_Category` — derived from the first two characters of `Product_Id` (`FD` food,
  `NC` non-consumable, `DR` drinks), which carries the signal the raw ID does not.
- `Store_Age_Years` — `2026 − Store_Establishment_Year`, so the model sees tenure rather than a
  calendar year it would treat as an arbitrary number.
- `Product_Sugar_Content` — `"reg"` folded into `"Regular"`, which were the same category recorded
  two ways.
- `Product_Id` and `Store_Establishment_Year` are dropped once their information has been extracted.

Numeric features are standardised and categoricals one-hot encoded with
`handle_unknown="ignore"`, all inside a single `Pipeline` so the identical transformation is applied
at training time and at inference time in the deployed app.

## Running it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r superkart_project/requirements.txt
export HF_TOKEN=...                         # a Hugging Face write token

python superkart_project/model_building/data_register.py
python superkart_project/model_building/prep.py
python superkart_project/model_building/train.py
python superkart_project/hosting/hosting_gradio.py
```

Experiment history:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## CI/CD

`pipeline.yml` runs four chained jobs on every push to `main` and on manual dispatch:

```
register-dataset → data-prep → model-training → deploy-hosting
```

The Hugging Face write token is supplied as the repository secret `HF_TOKEN`; the MLflow tracking
database is kept as a build artefact for 14 days.

## Note on hosting

The Streamlit/Docker front end in `deployment/` is the original target, but Hugging Face now limits
Docker Spaces to PRO accounts — `hosting.py` returns `402 Payment Required` on a free account. The
Gradio app in `deployment_gradio/` is functionally identical: same model, same eleven inputs, same
predictions.
