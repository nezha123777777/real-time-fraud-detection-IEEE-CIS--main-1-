"""End-to-end training pipeline for IEEE-CIS fraud detection.

Steps:
1) load and merge raw data
2) validate schema
3) clean
4) stateless features
5) chronological split
6) train-only imputers, encoders, aggregations
7) feature matrices
8) optional SMOTE on train only
9) optional anomaly pre-filter
10) optional Optuna tuning
11) train XGBoost
12) validate, threshold, explainability, tracking
13) optional test evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_data import load_train_data
from src.data.merge_data import merge_transaction_identity
from src.data.split_data import chronological_split
from src.data.validate_data import validate_merged_data
from src.features.aggregation_features import save_agg_maps
from src.features.build_features import (
    build_features_stateless,
    fit_stateful_features,
    transform_stateful_features,
)
from src.features.feature_selection import drop_constant_columns
from src.models.anomaly import save_isolation_forest, train_isolation_forest
from src.models.evaluate import evaluate_model
from src.models.explain import save_global_shap_importance
from src.models.onnx_utils import save_model_onnx
from src.models.thresholding import optimise_threshold, save_threshold
from src.models.tracking import log_training_run
from src.models.train import apply_smote, prepare_feature_matrix, save_model, train_xgboost
from src.models.tune import tune_xgboost
from src.preprocessing.categorical_encoding import FrequencyEncoder, save_encoders
from src.preprocessing.clean_data import clean_data
from src.preprocessing.missing_values import fit_imputers, save_imputers, transform_impute
from src.utils.config import load_config, resolve_path
from src.utils.io import load_parquet, save_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_clean_holdout_replay(
    cleaned_path: str | Path,
    holdout_transaction_ids: pd.Series,
    *,
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """Load cleaned data and return the chronologically sorted held-out replay rows."""
    cleaned_df = load_parquet(cleaned_path)
    if "TransactionID" not in cleaned_df.columns:
        raise KeyError("cleaned replay source must include TransactionID")

    holdout_ids = pd.Index(holdout_transaction_ids).dropna().unique()
    replay_df = cleaned_df[cleaned_df["TransactionID"].isin(holdout_ids)].copy()

    missing_ids = set(holdout_ids) - set(replay_df["TransactionID"])
    if missing_ids:
        raise ValueError(
            f"cleaned replay source is missing {len(missing_ids)} held-out TransactionID values"
        )

    return replay_df.sort_values(time_col).reset_index(drop=True)


def run_training_pipeline(evaluate_test: bool = False) -> dict[str, Any]:
    """Execute the full training pipeline."""
    paths_cfg = load_config("paths")
    model_cfg = load_config("model")
    target_col = model_cfg["target_column"]
    cleaned_train_path = resolve_path(paths_cfg["data"]["cleaned_train"])
    cleaned_holdout_replay_path = resolve_path(paths_cfg["data"]["cleaned_holdout_replay"])

    summary: dict[str, Any] = {}

    logger.info("=" * 60)
    logger.info("STEP 1 - Loading raw data")
    txn_df, idn_df = load_train_data()

    logger.info("=" * 60)
    logger.info("STEP 2 - Merging transaction and identity")
    merged = merge_transaction_identity(txn_df, idn_df)
    del txn_df, idn_df

    logger.info("=" * 60)
    logger.info("STEP 3 - Validating merged data")
    validate_merged_data(merged)

    logger.info("=" * 60)
    logger.info("STEP 4 - Cleaning")
    merged = clean_data(merged)
    save_parquet(merged, cleaned_train_path)

    logger.info("=" * 60)
    logger.info("STEP 5 - Stateless feature engineering")
    merged = build_features_stateless(merged)
    save_parquet(merged, resolve_path(paths_cfg["data"]["feature_ready_train"]))

    logger.info("=" * 60)
    logger.info("STEP 6 - Chronological split")
    train_df, valid_df, test_df = chronological_split(merged)
    del merged

    holdout_replay_df = build_clean_holdout_replay(
        cleaned_train_path,
        test_df["TransactionID"],
    )
    save_parquet(holdout_replay_df, cleaned_holdout_replay_path)
    summary["cleaned_holdout_replay_path"] = str(cleaned_holdout_replay_path)

    logger.info("=" * 60)
    logger.info("STEP 7 - Fitting preprocessing on train only")

    imputers = fit_imputers(train_df)
    save_imputers(imputers)
    train_df = transform_impute(train_df, imputers)
    valid_df = transform_impute(valid_df, imputers)
    test_df = transform_impute(test_df, imputers)

    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [target_col, "TransactionID"]]

    freq_encoder = FrequencyEncoder()
    if cat_cols:
        freq_encoder.fit(train_df, cat_cols)
        train_df = freq_encoder.transform(train_df)
        valid_df = freq_encoder.transform(valid_df)
        test_df = freq_encoder.transform(test_df)
    save_encoders({"frequency_encoder": freq_encoder})

    feature_maps = fit_stateful_features(train_df)
    save_agg_maps(feature_maps["agg_maps"])
    train_df = transform_stateful_features(train_df, feature_maps)
    valid_df = transform_stateful_features(valid_df, feature_maps)
    test_df = transform_stateful_features(test_df, feature_maps)

    train_df = drop_constant_columns(train_df)
    common_cols = [c for c in train_df.columns if c in valid_df.columns]
    valid_df = valid_df[common_cols]
    test_df = test_df[[c for c in common_cols if c in test_df.columns]]

    logger.info("=" * 60)
    logger.info("STEP 8 - Preparing feature matrices")
    y_train = train_df[target_col].values
    y_valid = valid_df[target_col].values
    y_test = test_df[target_col].values

    X_train = prepare_feature_matrix(train_df)
    X_valid = prepare_feature_matrix(valid_df)
    X_test = prepare_feature_matrix(test_df)

    save_parquet(X_train, resolve_path(paths_cfg["data"]["X_train"]))
    save_parquet(X_valid, resolve_path(paths_cfg["data"]["X_valid"]))
    save_parquet(X_test, resolve_path(paths_cfg["data"]["X_test"]))
    save_parquet(pd.DataFrame({target_col: y_train}), resolve_path(paths_cfg["data"]["y_train"]))
    save_parquet(pd.DataFrame({target_col: y_valid}), resolve_path(paths_cfg["data"]["y_valid"]))
    save_parquet(pd.DataFrame({target_col: y_test}), resolve_path(paths_cfg["data"]["y_test"]))

    feature_list_path = resolve_path(paths_cfg["artifacts"]["feature_lists_dir"]) / "features.txt"
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)
    feature_list_path.write_text("\n".join(X_train.columns.tolist()), encoding="utf-8")
    logger.info("Saved %d feature names -> %s", len(X_train.columns), feature_list_path)

    # ── Feast Integration (Optional but recommended) ──────────
    try:
        from datetime import datetime

        from feast import FeatureStore

        repo_path = resolve_path(".") / "feature_repo"
        if repo_path.exists():
            logger.info("Materializing features to Feast online store ...")
            store = FeatureStore(repo_path=str(repo_path))
            # Materialize the last year of data to online store
            store.materialize_incremental(end_date=datetime.now())
            logger.info("Feast materialization complete.")
    except Exception as e:
        logger.warning("Feast materialization skipped: %s", e)

    logger.info("=" * 60)
    logger.info("STEP 9 - Optional anomaly pre-filter")
    isolation_model = train_isolation_forest(X_train, y_train)
    if isolation_model is not None:
        isolation_model_path = save_isolation_forest(isolation_model)
        summary["isolation_model_path"] = str(isolation_model_path)

    logger.info("=" * 60)
    logger.info("STEP 10 - Optional Optuna tuning")
    params_override: dict[str, Any] | None = None
    model = None
    if model_cfg.get("tuning", {}).get("enabled", False):
        model, best_params, tuning_summary = tune_xgboost(X_train, y_train, X_valid, y_valid)
        params_override = best_params
        summary["best_params"] = best_params
        summary["tuning"] = tuning_summary

    logger.info("=" * 60)
    logger.info("STEP 11 - Training XGBoost")
    if model is None:
        X_train_fit, y_train_fit = apply_smote(X_train, y_train)
        model = train_xgboost(
            X_train_fit,
            y_train_fit,
            X_valid,
            y_valid,
            params_override=params_override,
        )
    native_model_path = save_model(model)
    onnx_model_path = save_model_onnx(model, feature_names=X_train.columns.tolist())
    summary["model_path"] = str(onnx_model_path)
    summary["native_model_path"] = str(native_model_path)

    logger.info("=" * 60)
    logger.info("STEP 12 - Evaluating on validation set")
    valid_metrics = evaluate_model(model, X_valid, y_valid, split_name="valid")
    summary["valid_metrics"] = valid_metrics

    thresh_info = optimise_threshold(y_valid, model.predict_proba(X_valid)[:, 1])
    save_threshold(thresh_info)
    summary["threshold"] = thresh_info

    explain_cfg = model_cfg.get("explainability", {})
    if explain_cfg.get("enabled", True) and explain_cfg.get("save_global_importance", True):
        shap_path = save_global_shap_importance(
            model,
            X_valid,
            split_name="valid",
            top_n=int(explain_cfg.get("top_n", 10)),
        )
        summary["shap_global_importance_path"] = str(shap_path)

    if evaluate_test:
        logger.info("=" * 60)
        logger.info("STEP 13 - Evaluating on test set")
        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
            split_name="test",
            threshold=thresh_info["threshold"],
        )
        summary["test_metrics"] = test_metrics

    fig_dir = resolve_path(paths_cfg["reports"]["figures_dir"])
    met_dir = resolve_path(paths_cfg["reports"]["metrics_dir"])

    mlflow_artifacts: list[str | Any] = [native_model_path, onnx_model_path, feature_list_path]
    if "shap_global_importance_path" in summary:
        mlflow_artifacts.append(summary["shap_global_importance_path"])

    # Automatically add all evaluation plots (.png)
    if fig_dir.exists():
        for fig_path in fig_dir.glob("*.png"):
            if str(fig_path) not in [str(p) for p in mlflow_artifacts]:
                mlflow_artifacts.append(str(fig_path))

    # Automatically add all metric reports (.json, .csv)
    if met_dir.exists():
        for met_path in met_dir.glob("*.[jc][sv]*"):
            if str(met_path) not in [str(p) for p in mlflow_artifacts]:
                mlflow_artifacts.append(str(met_path))

    mlflow_run_id = log_training_run(
        model=model,
        params={
            "xgboost": model_cfg.get("xgboost", {}),
            "smote": model_cfg.get("smote", {}),
            "isolation_forest": model_cfg.get("isolation_forest", {}),
            "thresholding": model_cfg.get("thresholding", {}),
            "best_params": summary.get("best_params", {}),
        },
        metrics={
            **summary["valid_metrics"],
            "threshold": summary["threshold"]["threshold"],
            "threshold_score": summary["threshold"]["score"],
            **summary.get("test_metrics", {}),
        },
        artifacts=mlflow_artifacts,
    )
    if mlflow_run_id:
        summary["mlflow_run_id"] = mlflow_run_id

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    return summary
