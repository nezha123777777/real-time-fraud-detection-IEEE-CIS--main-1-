"""Imputation helpers: fit on training data only, then transform other splits; defaults to artifacts/models/artifacts/imputers/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.config import resolve_path, load_config
from src.utils.io import save_artifact, load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame with all expected columns present."""
    missing_cols = [col for col in columns if col not in df.columns]
    if not missing_cols:
        return df
    missing_df = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
    return pd.concat([df, missing_df], axis=1)


def fit_imputers(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> dict[str, SimpleImputer]:
    """Fit separate imputers for numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        **Training** data only.
    numeric_strategy : str
        ``SimpleImputer`` strategy for numeric columns.
    categorical_strategy : str
        ``SimpleImputer`` strategy for categorical / object columns.

    Returns
    -------
    dict with keys ``"numeric"`` and ``"categorical"``, each holding
    a fitted ``SimpleImputer`` together with the column list.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    result: dict[str, dict] = {}

    if num_cols:
        imp_num = SimpleImputer(strategy=numeric_strategy)
        imp_num.fit(df[num_cols])
        result["numeric"] = {"imputer": imp_num, "columns": num_cols}
        logger.info("Fitted numeric imputer (%s) on %d columns.", numeric_strategy, len(num_cols))

    if cat_cols:
        imp_cat = SimpleImputer(strategy=categorical_strategy)
        imp_cat.fit(df[cat_cols].astype(str))
        result["categorical"] = {"imputer": imp_cat, "columns": cat_cols}
        logger.info("Fitted categorical imputer (%s) on %d columns.", categorical_strategy, len(cat_cols))

    return result


def transform_impute(df: pd.DataFrame, imputers: dict[str, dict]) -> pd.DataFrame:
    """Apply fitted imputers to a DataFrame (train, valid, or test).

    Parameters
    ----------
    df : pd.DataFrame
    imputers : dict
        Output of :func:`fit_imputers`.

    Returns
    -------
    pd.DataFrame  with missing values filled.
    """
    df = df.copy()

    if "numeric" in imputers:
        trained_cols = imputers["numeric"]["columns"]
        df = _ensure_columns(df, trained_cols)
        if trained_cols:
            df[trained_cols] = imputers["numeric"]["imputer"].transform(df[trained_cols])

    if "categorical" in imputers:
        trained_cols = imputers["categorical"]["columns"]
        df = _ensure_columns(df, trained_cols)
        if trained_cols:
            df[trained_cols] = imputers["categorical"]["imputer"].transform(
                df[trained_cols].astype(str)
            )

    remaining = df.isna().sum().sum()
    logger.info("After imputation — remaining NaNs: %d", remaining)
    return df


def save_imputers(imputers: dict[str, dict], directory: str | Path | None = None) -> Path:
    """Persist fitted imputers to disk.

    Parameters
    ----------
    imputers : dict
    directory : path, optional
        Defaults to ``artifacts/models/artifacts/imputers/``.

    Returns
    -------
    Path  — the saved file path.
    """
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["imputers_dir"])
    return save_artifact(imputers, Path(directory) / "imputers.joblib")


def load_imputers(directory: str | Path | None = None) -> dict[str, dict]:
    """Load previously fitted imputers from disk."""
    if directory is None:
        paths_cfg = load_config("paths")
        directory = resolve_path(paths_cfg["artifacts"]["imputers_dir"])
    return load_artifact(Path(directory) / "imputers.joblib")
