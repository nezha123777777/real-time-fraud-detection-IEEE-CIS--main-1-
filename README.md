# IEEE-CIS Fraud Detection

End-to-end machine learning pipeline for the Kaggle IEEE-CIS Fraud Detection dataset.

This repository keeps only the source code and project files needed to run the project. Kaggle data, trained models, logs, and generated reports stay local and are not committed.

## Project structure

```text
configs/      Configuration files
deployment/   FastAPI, Kafka, Streamlit, Docker setup
notebooks/    EDA, feature engineering, modeling, SHAP analysis
scripts/      Entry-point scripts
src/          Core pipeline code
tests/        Pytest test suite
```

## Requirements

- Python 3.10+
- `pip`
- Kaggle IEEE-CIS CSV files
- Docker only if you want to run the optional serving stack

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the Kaggle dataset files below and place them in `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

The pipeline creates validation and held-out test splits from those train files, so raw Kaggle test CSVs are not required.

## Run the project

Run the full training pipeline:

```powershell
python scripts/run_train.py
```

To enable SMOTE for class imbalance, set `smote.enabled: true` in `configs/model.yaml`.
The training pipeline now exports the serving model as `artifacts/models/final/xgb_final.onnx`.
Optional modules from the project plan are also available through config:
Isolation Forest, Optuna, MLflow, SHAP global importance, and API explainability.
The repo also includes a simple Evidently drift report script.

Run the full pipeline and also evaluate on the held-out test split:

```powershell
python scripts/run_train.py --evaluate-test
```

Run individual stages if needed:

```powershell
python scripts/run_data_prep.py
python scripts/run_feature_build.py
python scripts/run_evaluate.py
python scripts/run_drift_report.py
python scripts/run_inference.py
```

`python scripts/run_inference.py` scores the saved held-out split generated during training and writes `reports/tables/holdout_predictions.csv`.

## Tests

```powershell
pytest
```

## Optional local serving stack

If you want to run the API, Kafka consumer/producer, Postgres, dashboard, and MLflow locally:

```powershell
Copy-Item .env.example .env
cd deployment\docker
docker compose up --build
```

When running the full Docker Compose stack, services use Docker DNS names
internally. The compose file sets `POSTGRES_HOST=postgres` for the API,
consumer, and dashboard. If you run Streamlit directly on your host machine
instead of through Compose, keep `POSTGRES_HOST=localhost` in `.env`.

The FastAPI service uses ONNX Runtime for model inference. Preprocessing artefacts
(imputers, encoders, aggregation maps) remain Python artefacts on disk.
The API exposes `GET /health`, `POST /predict`, and `POST /predict/explain`.
The Docker stack also exposes MLflow at `http://localhost:5050`.
The Kafka producer now replays `data/interim/cleaned_holdout_replay.parquet` by
default so the dashboard shows held-out traffic instead of the pre-split
training feed.

To log local training runs to the Docker MLflow server:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5050"
python scripts/run_train.py
```

## Notes

- Model interpretation with SHAP is explored in `notebooks/05_error_analysis.ipynb`.
- Training outputs are written locally under `artifacts/` and `reports/`.
