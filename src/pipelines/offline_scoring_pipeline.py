"""Batch-score a processed feature file and save predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.onnx_utils import (
    align_expected_columns,
    load_expected_columns,
    load_onnx_session,
    predict_proba_onnx,
)
from src.models.thresholding import load_threshold
from src.utils.config import load_config, resolve_path
from src.utils.io import load_parquet, save_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_offline_scoring(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score a pre-processed feature file.

    Parameters
    ----------
    input_path : str or Path, optional
        Processed feature file with the same schema as ``X_test.parquet``.
        Defaults to the held-out split generated from the training data.
    output_path : str or Path, optional
        Where to save predictions. Supports ``.parquet`` and ``.csv``.
        Defaults to ``reports/tables/offline_scores.parquet``.
    """
    paths_cfg = load_config("paths")

    if input_path is None:
        input_path = resolve_path(paths_cfg["data"]["X_test"])
    if output_path is None:
        output_path = resolve_path(paths_cfg["reports"]["tables_dir"]) / "offline_scores.parquet"

    X = load_parquet(input_path)
    expected_cols = load_expected_columns(
        resolve_path(paths_cfg["artifacts"]["feature_lists_dir"]) / "features.txt"
    )
    X = align_expected_columns(X, expected_cols)
    model = load_onnx_session(resolve_path(paths_cfg["models"]["final_dir"]) / "xgb_final.onnx")
    thresh_info = load_threshold()

    probs = predict_proba_onnx(model, X)
    labels = (probs >= thresh_info["threshold"]).astype(int)

    result = pd.DataFrame({"fraud_prob": probs, "fraud_label": labels})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        result.to_csv(output_path, index=False)
    else:
        save_parquet(result, output_path)

    logger.info("Offline scoring complete → %s", output_path)
    return result
