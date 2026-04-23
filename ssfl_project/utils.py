"""Shared helper utilities — metrics, logging, file I/O."""
import json
import logging
import os
from typing import Dict, List

import numpy as np

import config


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    """Compute accuracy, macro F1, macro precision, macro recall, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    acc: float = float(accuracy_score(y_true, y_pred))
    f1: float = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec: float = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    rec: float = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    cm: np.ndarray = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    )
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "confusion_matrix": cm,
    }


def _json_safe(value):
    """Convert numpy scalars/arrays into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_round_metrics(
    round_num: int, metrics: dict, output_path: str
) -> None:
    """Append per-round metrics to a JSON log file on disk."""
    existing_data: List[dict] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_data = []
    entry: Dict = {"round": int(round_num), **_json_safe(metrics)}
    existing_data.append(entry)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)


def get_feature_column_names() -> List[str]:
    """Return the 115 canonical N-BaIoT feature column names.

    Ordering is grouped by time-decay level first (`L5`, `L3`, `L1`, `L0.1`,
    `L0.01`) so that flat index `j * 23 + k` corresponds to feature k within
    time window j — exactly what `reshape_sample_to_2d` expects.
    """
    return list(config.FEATURE_COLUMN_NAMES)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format across processes."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
