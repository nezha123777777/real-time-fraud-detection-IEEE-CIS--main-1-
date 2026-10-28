"""
scripts/run_data_prep.py
────────────────────────
Load, merge, validate, and clean the raw data.  Saves cleaned parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_train_data
from src.data.merge_data import merge_transaction_identity
from src.data.validate_data import validate_merged_data
from src.preprocessing.clean_data import clean_data
from src.utils.config import load_config, resolve_path
from src.utils.io import save_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    paths_cfg = load_config("paths")

    txn, idn = load_train_data()
    merged = merge_transaction_identity(txn, idn)
    validate_merged_data(merged)
    cleaned = clean_data(merged)
    save_parquet(cleaned, resolve_path(paths_cfg["data"]["cleaned_train"]))
    logger.info("Data prep complete.")


if __name__ == "__main__":
    main()
