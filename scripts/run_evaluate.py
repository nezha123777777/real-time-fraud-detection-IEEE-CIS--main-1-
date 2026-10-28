"""
scripts/run_evaluate.py
───────────────────────
Re-evaluate a saved model on a saved split.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import evaluate_model  # noqa: E402
from src.models.train import load_model  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.io import load_parquet  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    paths_cfg = load_config("paths")

    model = load_model(directory=resolve_path(paths_cfg["models"]["final_dir"]))
    X_valid = load_parquet(resolve_path(paths_cfg["data"]["X_valid"]))
    y_valid = load_parquet(resolve_path(paths_cfg["data"]["y_valid"]))["isFraud"].values

    metrics = evaluate_model(model, X_valid, y_valid, split_name="valid")
    print(metrics)


if __name__ == "__main__":
    main()
