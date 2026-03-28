"""
Shared utilities for the SSFL project.

Provides device detection, reproducibility seeding, evaluation metrics,
and serialisation helpers for Flower's ArrayRecord format.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ───────────────────────── device selection ───────────────────────


def get_device() -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ───────────────────────── reproducibility ────────────────────────


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────────────────────── evaluation metrics ─────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11,
) -> dict[str, float]:
    """
    Compute classification metrics matching the paper's evaluation protocol.

    Computes: Accuracy, macro-averaged Precision, Recall, and F1-Score
    (Equations 20–22 of the paper).

    Args:
        y_true: Ground truth labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        num_classes: Number of traffic classes.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 11,
) -> np.ndarray:
    """
    Compute the confusion matrix (for reproducing Fig. 3 of the paper).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
    """
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


# ───────────────────────── serialisation helpers ──────────────────


def labels_to_bytes(labels: np.ndarray) -> bytes:
    """
    Serialise a label array to bytes for Flower message transport.

    We use int16 since labels are in [-1, 10], saving bandwidth
    (the paper emphasises communication efficiency).
    """
    return labels.astype(np.int16).tobytes()


def bytes_to_labels(data: bytes) -> np.ndarray:
    """Deserialise a label array from bytes."""
    return np.frombuffer(data, dtype=np.int16).copy()
