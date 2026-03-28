from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import confusion_matrix, macro_f1_from_confusion


def train_local(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    total = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            n = yb.size(0)
            running_loss += float(loss.item()) * n
            total += n

    avg_loss = running_loss / max(total, 1)
    return {"train_loss": avg_loss}


def train_open_set_consistency(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    confidence_threshold: float,
    weight: float,
) -> dict[str, float]:
    """Run a simple pseudo-label consistency step on open data."""
    if weight <= 0.0:
        return {"ssfl_loss": 0.0, "ssfl_acceptance": 0.0}

    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    accepted = 0
    total = 0

    for xb, _ in loader:
        xb = xb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo = torch.max(probs, dim=1)
        mask = confidence >= confidence_threshold

        if int(mask.sum().item()) == 0:
            continue

        sel_logits = logits[mask]
        sel_pseudo = pseudo[mask]
        loss = criterion(sel_logits, sel_pseudo) * weight
        loss.backward()
        optimizer.step()

        n = int(mask.sum().item())
        running_loss += float(loss.item()) * n
        accepted += n
        total += xb.size(0)

    return {
        "ssfl_loss": running_loss / max(accepted, 1),
        "ssfl_acceptance": accepted / max(total, 1),
    }


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 11,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            preds = logits.argmax(dim=1)
            n = yb.size(0)

            running_loss += float(loss.item()) * n
            total += n
            correct += int((preds == yb).sum().item())
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "num_examples": 0,
        }

    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    cm = confusion_matrix(y_true_arr, y_pred_arr, num_classes=num_classes)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1_from_confusion(cm),
        "num_examples": total,
    }
