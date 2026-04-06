"""
Local training routines for SSFL.

Implements all six algorithmic steps from Algorithm 1 of the paper,
intentionally decoupled from Flower so each function can be tested
and debugged independently.

Step mapping:
    1. train_classifier()           — Eq. 11
    1b. compute_confidence_scores() — Eq. 12
    2. build_discriminator_dataset()— Eqs. 13–14
    2b. train_discriminator()       — train w^{k,d}
    3. filter_and_predict()         — Eq. 16
    5. distillation_train()         — Eq. 18
    (Step 4 is server-side voting — see server_app.py)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .model import Classifier, Discriminator


# ─────────────────────────────────────────────────────────────────
# Step 1: Train the classifier on private labelled data (Eq. 11)
# ─────────────────────────────────────────────────────────────────


def train_classifier(
    model: Classifier,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """
    Train the classifier w^{k,c} with the private labelled dataset D^k.

    This corresponds to the supervised training step in Algorithm 1,
    minimising cross-entropy loss (Eq. 11).

    Args:
        model: The classifier model.
        dataloader: DataLoader over private data ``(X^{k,c}, Y^{k,c})``.
        epochs: Number of local training epochs per communication round.
        lr: Learning rate (paper uses 0.0001).
        device: Compute device.

    Returns:
        Average training loss over the final epoch.
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        final_loss = epoch_loss / max(num_batches, 1)

    return final_loss


# ─────────────────────────────────────────────────────────────────
# Step 1 (cont.): Compute confidence scores on open data (Eq. 12)
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_confidence_scores(
    model: Classifier,
    X_open: torch.Tensor,
    device: torch.device,
    batch_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted labels and confidence scores on the open dataset D^o.

    Confidence c^{k,o}_j = max(F(x^o_j | w^{k,c}))  — Eq. 12.

    Args:
        model: Trained classifier.
        X_open: Open dataset features, shape ``(N_o, 1, 23, 5)``.
        device: Compute device.
        batch_size: Inference batch size.

    Returns:
        Tuple of:
        - ``predictions``: int array of shape ``(N_o,)`` — argmax class per sample.
        - ``confidences``: float array of shape ``(N_o,)`` — max softmax probability.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_confs = []

    for start in range(0, len(X_open), batch_size):
        batch = X_open[start : start + batch_size].to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_confs.append(confs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_confs)


# ─────────────────────────────────────────────────────────────────
# Step 2: Build discriminator training set (Eqs. 13–14)
# ─────────────────────────────────────────────────────────────────


def build_discriminator_dataset(
    confidences: np.ndarray,
    X_open: torch.Tensor,
    X_private: torch.Tensor,
    theta: float | None = None,
    batch_size: int = 100,
) -> DataLoader:
    """
    Build the discriminator training set D^{k,d}.

    - **Unfamiliar** (label [0,1]): open samples where confidence < θ  (Eq. 13)
    - **Familiar** (label [1,0]): all private samples                  (Eq. 14)

    The paper sets θ to the **median** of the client's confidence scores,
    which adapts per-client (Section V-F, Fig. 5).

    Args:
        confidences: Confidence scores from ``compute_confidence_scores()``.
        X_open: Open data features, shape ``(N_o, 1, 23, 5)``.
        X_private: Private data features, shape ``(N_k, 1, 23, 5)``.
        theta: Confidence threshold. If None, uses the median (paper default).
        batch_size: Batch size for the returned DataLoader.

    Returns:
        DataLoader yielding ``(X, y)`` where y ∈ {0 = familiar, 1 = unfamiliar}.
    """
    # Default θ = median of confidence scores (paper recommendation)
    if theta is None:
        theta = float(np.median(confidences))

    # --- Unfamiliar: open samples with confidence < θ ---
    unfamiliar_mask = confidences < theta
    X_unfamiliar = X_open[unfamiliar_mask]
    y_unfamiliar = torch.ones(X_unfamiliar.size(0), dtype=torch.long)  # label = 1

    # --- Familiar: all private samples ---
    y_familiar = torch.zeros(X_private.size(0), dtype=torch.long)  # label = 0

    # --- Concatenate ---
    X_disc = torch.cat([X_unfamiliar, X_private], dim=0)
    y_disc = torch.cat([y_unfamiliar, y_familiar], dim=0)

    dataset = TensorDataset(X_disc, y_disc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ─────────────────────────────────────────────────────────────────
# Step 2 (cont.): Train the discriminator
# ─────────────────────────────────────────────────────────────────


def train_discriminator(
    model: Discriminator,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """
    Train the discriminator w^{k,d} on the D^{k,d} dataset.

    Binary classification: familiar (0) vs. unfamiliar (1).

    Args:
        model: The discriminator model.
        dataloader: DataLoader from ``build_discriminator_dataset()``.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Compute device.

    Returns:
        Average training loss over the final epoch.
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        final_loss = epoch_loss / max(num_batches, 1)

    return final_loss


# ─────────────────────────────────────────────────────────────────
# Step 3: Filter predictions via discriminator (Eq. 16)
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def filter_and_predict(
    classifier: Classifier,
    discriminator: Discriminator,
    X_open: torch.Tensor,
    device: torch.device,
    batch_size: int = 100,
) -> np.ndarray:
    """
    Produce filtered hard-label predictions on the open dataset.

    For each sample x^o_j:
    - If discriminator says "familiar" (argmax = 0):
        prediction = argmax(classifier(x^o_j))
    - If discriminator says "unfamiliar" (argmax = 1):
        prediction = -1  (excluded from voting)

    This implements Eq. 16 of the paper.

    Args:
        classifier: Trained classifier model.
        discriminator: Trained discriminator model.
        X_open: Open dataset features, shape ``(N_o, 1, 23, 5)``.
        device: Compute device.
        batch_size: Inference batch size.

    Returns:
        Hard labels array of shape ``(N_o,)`` with values in
        ``{-1, 0, 1, ..., L-1}``.
    """
    classifier.to(device)
    classifier.eval()
    discriminator.to(device)
    discriminator.eval()

    all_labels: list[np.ndarray] = []

    for start in range(0, len(X_open), batch_size):
        batch = X_open[start : start + batch_size].to(device)

        # Classifier prediction
        cls_logits = classifier(batch)
        cls_preds = cls_logits.argmax(dim=1)  # shape (B,)

        # Discriminator decision
        disc_logits = discriminator(batch)
        disc_preds = disc_logits.argmax(dim=1)  # 0 = familiar, 1 = unfamiliar

        # Apply filter: unfamiliar → -1
        filtered = cls_preds.clone()
        filtered[disc_preds == 1] = -1

        all_labels.append(filtered.cpu().numpy())

    return np.concatenate(all_labels).astype(np.int16)


# ─────────────────────────────────────────────────────────────────
# Step 5: Distillation training with global hard labels (Eq. 18)
# ─────────────────────────────────────────────────────────────────


def distillation_train(
    model: Classifier,
    X_open: torch.Tensor,
    global_labels: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 100,
) -> float:
    """
    Train the classifier on the open data using global hard labels
    from the server (knowledge distillation step).

    The global hard labels P^s act as the "teacher" — the client's
    classifier learns to align its predictions on D^o with P^s (Eq. 18).

    Args:
        model: The classifier model to update.
        X_open: Open data features, shape ``(N_o, 1, 23, 5)``.
        global_labels: Server-aggregated hard labels, shape ``(N_o,)``.
            Values in ``{0, ..., L-1}``. Samples with label -1 from
            voting (if any) should be excluded before calling this.
        epochs: Number of distillation training epochs.
        lr: Learning rate.
        device: Compute device.
        batch_size: Training batch size.

    Returns:
        Average distillation loss over the final epoch.
    """
    model.to(device)
    model.train()

    # Filter out any invalid labels (should be rare after voting)
    valid_mask = global_labels >= 0
    X_valid = X_open[valid_mask]
    y_valid = torch.from_numpy(global_labels[valid_mask].astype(np.int64))

    if len(X_valid) == 0:
        return 0.0

    dataset = TensorDataset(X_valid, y_valid)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        final_loss = epoch_loss / max(num_batches, 1)

    return final_loss


# ─────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model: Classifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate a classifier on a test DataLoader.

    Args:
        model: Trained classifier.
        dataloader: Test DataLoader yielding ``(X, y)`` tuples.
        device: Compute device.

    Returns:
        Tuple of ``(loss, y_true, y_pred)`` as NumPy arrays.
    """
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_true, all_pred = [], []
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        num_batches += 1
        all_true.append(y_batch.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, np.concatenate(all_true), np.concatenate(all_pred)
