"""Inference pipeline: load saved artefacts and score new data without retraining."""

from __future__ import annotations

import pandas as pd

from src.data.merge_data import merge_transaction_identity
from src.features.aggregation_features import load_agg_maps
from src.features.build_features import build_features_stateless, transform_stateful_features
from src.models.onnx_utils import (
    align_expected_columns,
    load_expected_columns,
    load_onnx_session,
    predict_proba_onnx,
)
from src.models.thresholding import load_threshold
from src.models.train import prepare_feature_matrix
from src.preprocessing.categorical_encoding import load_encoders
from src.preprocessing.clean_data import clean_data
from src.preprocessing.missing_values import load_imputers, transform_impute
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)   


def run_inference(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score new transactions using saved artefacts.

    Returns a DataFrame with columns: ``TransactionID``, ``fraud_prob``,
    ``fraud_label``.
    """
    paths_cfg = load_config("paths")

    # Merge
    merged = merge_transaction_identity(transaction_df, identity_df)

"""Inference pipeline: load saved artefacts and score new data without retraining."""

from __future__ import annotations

import pandas as pd

from src.data.merge_data import merge_transaction_identity
from src.features.aggregation_features import load_agg_maps
from src.features.build_features import build_features_stateless, transform_stateful_features
from src.models.onnx_utils import (
    align_expected_columns,
    load_expected_columns,
    load_onnx_session,
    predict_proba_onnx,
)
from src.models.thresholding import load_threshold
from src.models.train import prepare_feature_matrix
from src.preprocessing.categorical_encoding import load_encoders
from src.preprocessing.clean_data import clean_data
from src.preprocessing.missing_values import load_imputers, transform_impute
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_inference(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score new transactions using saved artefacts.

    Returns a DataFrame with columns: ``TransactionID``, ``fraud_prob``,
    ``fraud_label``.
    """
    paths_cfg = load_config("paths")

    # Merge
    merged = merge_transaction_identity(transaction_df, identity_df)

    # Clean
    merged = clean_data(merged)

    # Stateless features
    merged = build_features_stateless(merged)

    # Load & apply saved preprocessing
    imputers = load_imputers()
    merged = transform_impute(merged, imputers)

    encoders = load_encoders()
    freq_enc = encoders.get("frequency_encoder")
    if freq_enc:
        merged = freq_enc.transform(merged)

    agg_maps = load_agg_maps()
    merged = transform_stateful_features(merged, {"agg_maps": agg_maps})

    # ── Feast Online Retrieval Integration ──────────
    try:
        from feast import FeatureStore
        repo_path = resolve_path(".") / "feature_repo"
        if repo_path.exists():
            store = FeatureStore(repo_path=str(repo_path))
            # Define entities to fetch features for (the ones in the current dataframe)
            entity_rows = [{"TransactionID": tid} for tid in merged["TransactionID"]]
            
            logger.info("Fetching online features from Feast for %d keys ...", len(entity_rows))
            # Fetch a subset of features to demonstrate the lookup
            feast_features = store.get_online_features(
                features=[
                    "transaction_stats:TransactionAmt",
                    "transaction_stats:hour",
                    "card_info:card2",
                ],
                entity_rows=entity_rows
            ).to_dict()
            
            # Note: In a full migration, we would merge/replace columns here.
            # For now, we log that we successfully retrieved them.
            if "TransactionAmt" in feast_features and feast_features["TransactionAmt"][0] is not None:
                logger.info("Successfully retrieved Feast features for TransactionID: %s", entity_rows[0]["TransactionID"])
    except Exception as e:
        logger.warning("Feast online retrieval skipped: %s", e)

    # Prepare features
    txn_ids = merged["TransactionID"].copy()
    X = prepare_feature_matrix(merged)
    expected_cols = load_expected_columns(
        resolve_path(paths_cfg["artifacts"]["feature_lists_dir"]) / "features.txt"
    )
    X = align_expected_columns(X, expected_cols)

    # Load model & threshold
    model = load_onnx_session(resolve_path(paths_cfg["models"]["final_dir"]) / "xgb_final.onnx")
    thresh_info = load_threshold()
    threshold = thresh_info["threshold"]

    # Predict
    probs = predict_proba_onnx(model, X)
    labels = (probs >= threshold).astype(int)

    result = pd.DataFrame({
        "TransactionID": txn_ids,
        "fraud_prob": probs,
        "fraud_label": labels,
    })

    logger.info(
        "Inference complete — %d transactions, %d flagged as fraud (%.2f%%)",
        len(result), int(labels.sum()), 100 * labels.mean(),
    )
    return result
