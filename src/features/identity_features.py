"""Identity/device feature helpers (browser family, mobile flag, screen resolution)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_browser_features(df: pd.DataFrame, col: str = "id_31") -> pd.DataFrame:
    """Parse ``id_31`` (browser string) into simplified features.

    New columns
    -----------
    ``browser_family``  — top-level browser name (chrome, firefox, …)
    ``is_mobile_browser`` — 1 if the browser string suggests a mobile UA

    These are stateless transforms (no fitting required).
    """
    df = df.copy()

    if col not in df.columns:
        logger.warning("Column '%s' not found — skipping browser features.", col)
        return df

    browser_str = df[col].fillna("unknown").astype(str).str.lower()

    # Extract family
    families = ["chrome", "firefox", "safari", "edge", "opera", "ie", "samsung"]
    df["browser_family"] = "other"
    for fam in families:
        mask = browser_str.str.contains(fam, na=False)
        df.loc[mask, "browser_family"] = fam

    # Mobile flag
    mobile_keywords = ["mobile", "android", "iphone", "sm-", "samsung"]
    df["is_mobile_browser"] = browser_str.apply(
        lambda s: int(any(kw in s for kw in mobile_keywords))
    ).astype("int8")

    logger.info("Added browser features: browser_family, is_mobile_browser")
    return df


def add_screen_resolution_features(df: pd.DataFrame, col: str = "id_33") -> pd.DataFrame:
    """Parse ``id_33`` (screen resolution like ``"1920x1080"``) into numeric features.

    New columns
    -----------
    ``screen_width``   — pixel width
    ``screen_height``  — pixel height
    ``screen_area``    — width × height (proxy for device class)
    """
    df = df.copy()

    if col not in df.columns:
        logger.warning("Column '%s' not found — skipping screen resolution features.", col)
        return df

    # Parse "WxH" strings
    parts = df[col].astype(str).str.split("x", expand=True)
    if parts.shape[1] >= 2:
        df["screen_width"] = pd.to_numeric(parts[0], errors="coerce").astype("float32")
        df["screen_height"] = pd.to_numeric(parts[1], errors="coerce").astype("float32")
        df["screen_area"] = (df["screen_width"] * df["screen_height"]).astype("float32")
    else:
        df["screen_width"] = np.nan
        df["screen_height"] = np.nan
        df["screen_area"] = np.nan

    logger.info("Added screen resolution features: screen_width, screen_height, screen_area")
    return df


def add_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all identity-based feature creation.

    Calls :func:`add_browser_features` and
    :func:`add_screen_resolution_features`.
    """
    df = add_browser_features(df)
    df = add_screen_resolution_features(df)
    return df
