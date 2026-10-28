"""Load YAML config files and resolve paths relative to the project root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _find_project_root() -> Path:
    """Return the project root directory.

    Resolution order:
    1. ``PROJECT_ROOT`` environment variable (set in ``.env``).
    2. Walk up from this file until a directory containing ``configs/`` is found.
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not locate project root. Set PROJECT_ROOT or ensure 'configs/' exists."
    )


# Load .env as early as possible
load_dotenv(_find_project_root() / ".env", override=False)

PROJECT_ROOT: Path = _find_project_root()


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from ``configs/<name>.yaml``.

    Parameters
    ----------
    name : str
        Config filename **without** the ``.yaml`` extension,
        e.g. ``"paths"``, ``"model"``.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    config_path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str) -> Path:
    """Resolve a path string from ``paths.yaml`` against the project root.

    Parameters
    ----------
    relative : str
        Path string relative to the project root, e.g.
        ``"data/raw/train_transaction.csv"``.

    Returns
    -------
    Path
        Absolute ``Path`` object.
    """
    return (PROJECT_ROOT / relative).resolve()
