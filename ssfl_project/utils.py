"""Shared helper utilities — logging, generic file I/O, and a thin backward-
compatible `compute_metrics` wrapper over `metrics.compute_classification_metrics`.

The authoritative classification math lives in `metrics.py`. This module's
job is only the *side-effect* stuff — logging setup, JSON persistence, the
feature-name lookup — so that `metrics.py` stays a pure, testable
compute-only module.
"""
import json
import logging
import os
from typing import Dict, List

import numpy as np

import config


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    """Backward-compatible facade over `metrics.compute_classification_metrics`.

    Returns the same keys as before (`accuracy`, `f1_macro`, `precision_macro`,
    `recall_macro`, `confusion_matrix`) so older tests / call sites don't
    break, but also forwards the richer keys (`f1_weighted`, `f1_per_class`,
    per-class precision/recall/support, `class_names`) that our Table II
    analogue needs. The confusion matrix field matches the legacy shape
    (numpy array) for backward compatibility.

    Prefer calling `metrics.compute_classification_metrics` directly in
    new code; this wrapper exists only so existing call sites keep working.
    """
    from metrics import compute_classification_metrics

    result: dict = compute_classification_metrics(y_true, y_pred, num_classes)
    # Legacy shape: confusion matrix as np.ndarray, not list-of-lists.
    cm_list = result.get("confusion_matrix")
    result["confusion_matrix"] = (
        np.asarray(cm_list, dtype=np.int64) if cm_list is not None else None
    )
    return result


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
