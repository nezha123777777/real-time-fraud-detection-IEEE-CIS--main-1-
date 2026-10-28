"""Generate a simple Evidently drift report from saved feature splits."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.io import load_parquet  # noqa: E402


def main() -> None:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("Evidently is not installed. Install it to generate drift reports.")
        sys.exit(1)

    paths_cfg = load_config("paths")
    baseline = load_parquet(resolve_path(paths_cfg["data"]["X_train"]))
    current = load_parquet(resolve_path(paths_cfg["data"]["X_test"]))

    report = Report([DataDriftPreset()])
    report.run(reference_data=baseline, current_data=current)

    out_dir = resolve_path(paths_cfg["reports"]["metrics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evidently_drift_report.html"
    report.save_html(str(out_path))
    print(f"Saved drift report -> {out_path}")


if __name__ == "__main__":
    main()
