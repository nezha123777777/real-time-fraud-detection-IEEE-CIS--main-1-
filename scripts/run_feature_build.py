"""
scripts/run_feature_build.py
────────────────────────────
Build features on the cleaned dataset and save as parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features_stateless
from src.utils.config import load_config, resolve_path
from src.utils.io import load_parquet, save_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    paths_cfg = load_config("paths")
    cleaned = load_parquet(resolve_path(paths_cfg["data"]["cleaned_train"]))
    featured = build_features_stateless(cleaned)
    save_parquet(featured, resolve_path(paths_cfg["data"]["feature_ready_train"]))
    logger.info("Feature build complete.")


if __name__ == "__main__":
    main()
