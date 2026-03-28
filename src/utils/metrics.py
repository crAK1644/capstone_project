from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    f1_scores: list[float] = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1_scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(f1_scores))
