"""Configuration loader and management utilities."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path relative to project root or absolute path.

    Returns:
        Dictionary containing configuration parameters.
    """
    root = get_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
