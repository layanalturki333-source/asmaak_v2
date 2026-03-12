"""
I/O utilities for labels and config.

Handles loading/saving label mappings (index <-> Arabic sign label)
so that training and inference use the same vocabulary.
"""

import json
from pathlib import Path
from typing import List, Optional


def load_labels(path: Path) -> List[str]:
    """
    Load label list from JSON file.
    File should contain a JSON array of strings, e.g. ["سلام", "شكراً", ...].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Labels file must contain a JSON array of strings")
    return [str(x) for x in data]


def save_labels(labels: List[str], path: Path) -> None:
    """Save label list to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
