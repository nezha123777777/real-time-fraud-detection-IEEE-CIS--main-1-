"""
scripts/run_inference.py
------------------------
Score the saved held-out split using artefacts generated from the train files.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.offline_scoring_pipeline import run_offline_scoring
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    output_path = PROJECT_ROOT / "reports" / "tables" / "holdout_predictions.csv"
    result = run_offline_scoring(output_path=output_path)
    logger.info("Predictions saved to %s", output_path)
    print(result.head(10))


if __name__ == "__main__":
    main()
