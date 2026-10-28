"""
scripts/run_train.py
────────────────────
CLI entry-point for the full training pipeline.

Usage
-----
    python scripts/run_train.py                # dev run (no test eval)
    python scripts/run_train.py --evaluate-test  # final run with test eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.training_pipeline import run_training_pipeline  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="IEEE-CIS Fraud Detection — Training Pipeline")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help="Also evaluate on the held-out test set (final run only).",
    )
    args = parser.parse_args()

    logger.info("Starting training pipeline (evaluate_test=%s)", args.evaluate_test)

    try:
        summary = run_training_pipeline(evaluate_test=args.evaluate_test)
    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    if "valid_metrics" in summary:
        vm = summary["valid_metrics"]
        print(f"  Validation ROC-AUC : {vm['roc_auc']:.4f}")
        print(f"  Validation PR-AUC  : {vm['pr_auc']:.4f}")
        print(f"  Validation Recall  : {vm['recall']:.4f}")
        print(f"  Validation F1      : {vm['f1']:.4f}")

    if "threshold" in summary:
        t = summary["threshold"]
        print(f"  Optimal threshold  : {t['threshold']:.3f} ({t['metric']}={t['score']:.4f})")

    if "test_metrics" in summary:
        tm = summary["test_metrics"]
        print(f"  Test ROC-AUC       : {tm['roc_auc']:.4f}")
        print(f"  Test PR-AUC        : {tm['pr_auc']:.4f}")
        print(f"  Test Recall        : {tm['recall']:.4f}")
        print(f"  Test F1            : {tm['f1']:.4f}")

    if "model_path" in summary:
        print(f"  Model saved to     : {summary['model_path']}")
    if "native_model_path" in summary:
        print(f"  Native model saved : {summary['native_model_path']}")
    if "isolation_model_path" in summary:
        print(f"  Isolation model    : {summary['isolation_model_path']}")
    if "shap_global_importance_path" in summary:
        print(f"  SHAP report        : {summary['shap_global_importance_path']}")
    if "mlflow_run_id" in summary:
        print(f"  MLflow run id      : {summary['mlflow_run_id']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
