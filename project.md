# IEEE-CIS Fraud Detection — Project Documentation

> **Goal**: Build a production-ready, end-to-end machine learning pipeline to detect fraudulent transactions from the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Tools](#2-tech-stack--tools)
3. [ML Pipeline Steps](#3-ml-pipeline-steps)
4. [Directory Structure](#4-directory-structure)
5. [Configuration System](#5-configuration-system)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Model Performance](#7-model-performance)
8. [Experiment Tracking (MLflow)](#8-experiment-tracking-mlflow)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)

---

## 1. Project Overview

The project uses the training portion of the Kaggle dataset, split across two CSV files:

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train_transaction.csv` | 590,540 | 394 | Core transaction data |
| `train_identity.csv` | 144,233 | 41 | Device / identity metadata |

The two tables are joined on `TransactionID` (left join → 24.4 % identity match rate).  
**Target**: `isFraud` — highly imbalanced at ~3.5% positive rate.

---

## 2. Tech Stack & Tools

### Core ML & Data

| Tool | Version | Role |
|------|---------|------|
| **Python** | 3.11 | Runtime |
| **pandas** | ≥2.0 | Data manipulation, CSV/Parquet I/O |
| **NumPy** | ≥1.24 | Numerical operations |
| **scikit-learn** | ≥1.3 | Imputers, encoders, metrics, train/test split |
| **XGBoost** | ≥2.0 | Primary gradient-boosted tree model |
| **LightGBM** | ≥4.0 | Optional second model |
| **imbalanced-learn** | ≥0.11 | SMOTE oversampling (optional) |
| **joblib** | ≥1.3 | Artifact serialisation (imputers, encoders) |

### Explainability

| Tool | Version | Role |
|------|---------|------|
| **SHAP** | ≥0.43 | Global feature importance, per-prediction explanations |

### Hyperparameter Tuning

| Tool | Version | Role |
|------|---------|------|
| **Optuna** | ≥3.6 | Bayesian HPO with TPE sampler + Median pruner |

### Anomaly Detection

| Tool | Version | Role |
|------|---------|------|
| **scikit-learn IsolationForest** | ≥1.3 | Optional pre-filter to flag anomalous transactions |

### Model Export & Serving

| Tool | Version | Role |
|------|---------|------|
| **ONNX** | ≥1.15 | Model interchange format for runtime-agnostic inference |
| **onnxmltools** | ≥1.16 | Converts XGBoost → ONNX |
| **onnxruntime** | ≥1.17 | Runs ONNX model in production (no XGBoost dependency needed) |

### Experiment Tracking

| Tool | Version | Role |
|------|---------|------|
| **MLflow** | ≥3.0 | Logs params, metrics, XGBoost model, ONNX, SHAP plots per run |

### Data Drift & Monitoring

| Tool | Version | Role |
|------|---------|------|
| **Evidently** | ≥0.4 | Data drift reports between training and production distributions |

### API & Streaming

| Tool | Version | Role |
|------|---------|------|
| **FastAPI** | ≥0.103 | REST inference API (`/predict`, `/predict/explain`, `/health`) |
| **Uvicorn** | ≥0.23 | ASGI server for FastAPI |
| **kafka-python** | ≥2.0 | Consumes real-time transaction events from Kafka |
| **SQLAlchemy** | ≥2.0 | ORM for prediction logging to PostgreSQL |
| **psycopg2** | ≥2.9 | PostgreSQL driver |

### Dashboard

| Tool | Version | Role |
|------|---------|------|
| **Streamlit** | ≥1.37 | Interactive monitoring dashboard |

### Configuration & I/O

| Tool | Version | Role |
|------|---------|------|
| **PyYAML** | ≥6.0 | YAML config files (`configs/*.yaml`) |
| **python-dotenv** | ≥1.0 | Loads secrets from `.env` |
| **PyArrow** | ≥14.0 | Parquet read/write for feature matrices |
| **httpx** | ≥0.27 | Async HTTP client (used in inference pipeline) |

### Visualisation

| Tool | Version | Role |
|------|---------|------|
| **Matplotlib** | ≥3.7 | PR curve, ROC curve, confusion matrix plots |
| **Seaborn** | ≥0.13 | Feature importance and distribution plots |

### Notebooks

| Tool | Version | Role |
|------|---------|------|
| **JupyterLab** | ≥4.0 | EDA, experimentation notebooks |
| **ipykernel** | ≥6.0 | Python kernel for Jupyter |

### Testing & Code Quality

| Tool | Version | Role |
|------|---------|------|
| **pytest** | ≥7.4 | Unit and integration tests |
| **pytest-cov** | ≥4.1 | Code coverage reports |
| **Ruff** | ≥0.1 | Fast Python linter |
| **Black** | ≥23.0 | Code formatter |

### Infrastructure

| Tool | Role |
|------|------|
| **Docker** | Containerises the FastAPI serving image |
| **Git** | Version control |

---

## 3. ML Pipeline Steps

The full pipeline is orchestrated by `scripts/run_train.py` → `src/pipelines/training_pipeline.py`.

```
STEP 1  Load raw CSVs (transaction + identity)
STEP 2  Merge on TransactionID (left join)
STEP 3  Validate merged data (schema, target distribution, ID uniqueness)
STEP 4  Clean (drop high-missing columns >50%, dtype fixes)
STEP 5  Stateless feature engineering
          - Time features: hour, day_of_week, is_weekend, is_night
          - Card velocity: time_since_last_card1
          - Amount features: log_TransactionAmt, decimal part, is_round
          - D-column deltas
STEP 6  Chronological split  (70% train / 15% valid / 15% test)
STEP 7  Fit preprocessing on train only (prevent leakage)
          - Median imputer  → numeric columns
          - Most-frequent imputer → categorical columns
          - FrequencyEncoder → categorical columns
          - Aggregation features (7 group-by statistics)
          - Drop constant columns
STEP 8  Save feature matrices as Parquet
STEP 9  Optional Isolation Forest anomaly pre-filter
STEP 10 Optional Optuna hyperparameter tuning (25 trials, TPE + MedianPruner)
STEP 11 Train XGBoost (scale_pos_weight to handle imbalance, early stopping)
          - Export native JSON model
          - Export ONNX model
STEP 12 Evaluate on validation set
          - Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1
          - Optimal threshold search (maximise F1)
          - SHAP global importance
          - Plots: PR curve, ROC curve, confusion matrix, feature importance
STEP 13 Log run to MLflow (params, metrics, model, artifacts)
```

---

## 4. Directory Structure

```
ieee_fraud_project/
├── configs/                  # YAML configuration files
│   ├── data.yaml             # Data paths and thresholds
│   ├── model.yaml            # Model hyperparams, MLflow, SMOTE, tuning
│   └── paths.yaml            # File path aliases
│
├── data/
│   ├── raw/                  # Original CSVs (not tracked by git)
│   ├── interim/              # Cleaned & feature-ready Parquets
│   └── processed/            # Final X_train/valid/test + y splits
│
├── artifacts/
│   └── models/
│       ├── final/            # xgb_final.json, xgb_final.onnx
│       └── artifacts/
│           ├── imputers/     # imputers.joblib
│           ├── encoders/     # encoders.joblib
│           ├── feature_lists/# features.txt, agg_maps.joblib
│           └── thresholds/   # threshold.json
│
├── reports/
│   ├── metrics/              # metrics_valid.json, shap_global_importance.csv
│   └── figures/              # PR curve, ROC curve, confusion matrix, SHAP plot
│
├── src/
│   ├── data/                 # Data loading utilities
│   ├── preprocessing/        # Missing value imputation, categorical encoding
│   ├── features/             # Stateless & aggregation feature engineering
│   ├── models/               # XGBoost training, tracking (MLflow)
│   ├── pipelines/            # training_pipeline.py, inference_pipeline.py
│   └── utils/                # Config loader, logger, metrics helpers
│
├── scripts/
│   ├── run_train.py          # End-to-end training entry point
│   ├── run_evaluate.py       # Standalone evaluation script
│   ├── run_data_prep.py      # Data preparation only
│   └── run_feature_build.py  # Feature engineering only
│
├── deployment/
│   ├── dashboard/app.py      # Streamlit monitoring dashboard
│   ├── Dockerfile            # Docker image for FastAPI server
│   └── requirements_serving.txt
│
├── tests/                    # pytest unit & integration tests
├── notebooks/                # EDA and experimentation notebooks
├── requirements.txt          # Full dependency list
├── pyproject.toml            # Ruff + Black config
└── mlflow.db                 # Local MLflow SQLite store
```

---

## 5. Configuration System

All behaviour is controlled through YAML files in `configs/`. No hardcoded values in source code.

| Config File | Controls |
|-------------|---------|
| `configs/data.yaml` | Column lists, missing-value thresholds, validation rules |
| `configs/model.yaml` | XGBoost params, SMOTE, Isolation Forest, Optuna, MLflow, thresholding, evaluation |
| `configs/paths.yaml` | All file/directory paths used across the pipeline |

Loaded via `src/utils/config.py → load_config(name)`.  
Secrets (DB passwords, Kafka brokers) are in `.env` and loaded via `python-dotenv`.

---

## 6. Running the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full training pipeline
python scripts/run_train.py

# 3. Evaluate on test set
python scripts/run_train.py --evaluate-test

# 4. Feature engineering only
python scripts/run_feature_build.py

# 5. Data preparation only
python scripts/run_data_prep.py

# 6. Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 5050
# → open http://localhost:5050

# 7. Launch Streamlit dashboard
streamlit run deployment/dashboard/app.py
```

---

## 7. Model Performance

Results on the **validation set** (chronological split, 15% of data):

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9250** |
| PR-AUC | **0.5835** |
| Precision | 0.6400 |
| Recall | 0.5108 |
| F1 | 0.5682 |
| Optimal threshold | 0.650 |

> Trained on 413,378 transactions — `scale_pos_weight = 27.43` to handle the 3.5 % fraud rate.  
> Best XGBoost iteration: 983 / 1000 (early stopping round 50).

---

## 8. Experiment Tracking (MLflow)

MLflow tracking is configured in `configs/model.yaml` under the `mlflow:` key.

| Setting | Default |
|---------|---------|
| `enabled` | `true` |
| `tracking_uri` | local SQLite (`mlflow.db`) |
| `experiment_name` | `ieee-fraud-detection` |
| `run_name` | `xgboost_smote_onnx` |

**Per run, MLflow logs:**
- All XGBoost hyperparameters (flattened)
- Evaluation metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
- XGBoost native model
- ONNX model file
- Feature list
- SHAP global importance plot

**Access the UI:**
```bash
mlflow ui --host 0.0.0.0 --port 5050
```

---

## 9. Deployment

### FastAPI Inference Server
- Endpoint `POST /predict` — accepts transaction JSON, returns fraud probability + label
- Endpoint `POST /predict/explain` — returns SHAP explanation for a prediction
- Endpoint `GET /health` — liveness probe

### Docker
```bash
docker build -f deployment/Dockerfile -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### Streamlit Dashboard
Real-time monitoring of:
- Incoming prediction volumes
- Fraud rate over time
- Data drift reports (Evidently)
- Model performance indicators

---

## 10. Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

Test modules:

| File | Covers |
|------|--------|
| `test_evaluation.py` | Metrics computation, threshold search |
| `test_training_pipeline.py` | End-to-end pipeline integration |
| `test_feature_engineering.py` | Stateless & aggregation features |
| `test_missing_values.py` | Imputation logic |
| `test_validate_data.py` | Schema and target validation |
| `conftest.py` | Shared fixtures (sample DataFrames, configs) |
