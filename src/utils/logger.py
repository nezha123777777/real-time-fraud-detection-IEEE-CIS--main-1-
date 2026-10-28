"""Configure logging from configs/logging.yaml."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from src.utils.config import PROJECT_ROOT


_CONFIGURED = False


def _setup_logging() -> None:
    """Read ``configs/logging.yaml`` and apply it to the stdlib logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    config_path = PROJECT_ROOT / "configs" / "logging.yaml"
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            log_cfg = yaml.safe_load(fh)
        # Make the file handler path absolute
        if "handlers" in log_cfg and "file" in log_cfg["handlers"]:
            log_cfg["handlers"]["file"]["filename"] = str(
                PROJECT_ROOT / log_cfg["handlers"]["file"]["filename"]
            )
        logging.config.dictConfig(log_cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the system on first call.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    """
    _setup_logging()
    return logging.getLogger(name)
