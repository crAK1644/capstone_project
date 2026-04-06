"""
Data loading utilities for the SSFL project.

Wraps the pre-processed ``prepared_data/`` directory (produced by
``prepare_dataset.py``) and provides clean PyTorch DataLoader objects
for FL clients, the shared open dataset, and the held-out test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ───────────────────────── path constants ─────────────────────────

DATA_ROOT = Path(__file__).resolve().parent.parent / "prepared_data"


# ───────────────────────── low-level loaders ──────────────────────


def _load_npy_as_tensor(path: Path, dtype: torch.dtype) -> torch.Tensor:
    """Load a ``.npy`` file and convert to a PyTorch tensor."""
    arr = np.load(path)
    return torch.from_numpy(arr).to(dtype)


# ───────────────────────── public API ─────────────────────────────


def load_split(
    split: str,
    *,
    use_2d: bool = True,
    batch_size: int = 100,
    shuffle: bool = False,
) -> DataLoader:
    """
    Load a global data split (``private``, ``open``, or ``test``) as a DataLoader.

    Args:
        split: One of ``"private"``, ``"open"``, ``"test"``.
        use_2d: If True, load the CNN-ready ``X_2d.npy`` (N, 1, 23, 5).
                If False, load flat features ``X.npy`` (N, 115).
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.

    Returns:
        A PyTorch DataLoader yielding ``(X, y)`` tuples.
    """
    split_dir = DATA_ROOT / split
    x_file = "X_2d.npy" if use_2d else "X.npy"

    X = _load_npy_as_tensor(split_dir / x_file, torch.float32)
    y = _load_npy_as_tensor(split_dir / "y.npy", torch.long)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_open_data_tensors(*, use_2d: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the open dataset as raw tensors (not wrapped in a DataLoader).

    This is useful when we need direct array access for building the
    discriminator dataset or serialising predictions.

    Returns:
        ``(X_open, y_open)`` — note: ``y_open`` labels exist in the dataset
        but are **not used** during SSFL training (unlabeled setting).
    """
    split_dir = DATA_ROOT / "open"
    x_file = "X_2d.npy" if use_2d else "X.npy"

    X = _load_npy_as_tensor(split_dir / x_file, torch.float32)
    y = _load_npy_as_tensor(split_dir / "y.npy", torch.long)
    return X, y


def load_client_data(
    scenario: int,
    client_id: int,
    *,
    batch_size: int = 100,
    shuffle: bool = True,
) -> DataLoader:
    """
    Load a specific client's private data for a given scenario.

    Args:
        scenario: Scenario number (1, 2, or 3).
        client_id: Client index within the scenario.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle.

    Returns:
        A PyTorch DataLoader yielding ``(X, y)`` tuples.
    """
    client_dir = DATA_ROOT / f"scenario_{scenario}" / f"client_{client_id}"

    X = _load_npy_as_tensor(client_dir / "X_2d.npy", torch.float32)
    y = _load_npy_as_tensor(client_dir / "y.npy", torch.long)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_client_data_tensors(
    scenario: int,
    client_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a client's private data as raw tensors (no DataLoader wrapping).

    Useful for building the discriminator's familiar-sample set.
    """
    client_dir = DATA_ROOT / f"scenario_{scenario}" / f"client_{client_id}"

    X = _load_npy_as_tensor(client_dir / "X_2d.npy", torch.float32)
    y = _load_npy_as_tensor(client_dir / "y.npy", torch.long)
    return X, y


def load_scenario_summary(scenario: int) -> list[dict]:
    """
    Load the metadata summary for a scenario.

    Returns:
        List of dicts, each with keys:
        ``client_id``, ``device_id``, ``num_samples``, ``labels_present``.
    """
    summary_path = DATA_ROOT / f"scenario_{scenario}" / "summary.json"
    with open(summary_path) as f:
        return json.load(f)


def get_num_clients(scenario: int) -> int:
    """Return the number of clients in a given scenario."""
    summary = load_scenario_summary(scenario)
    return len(summary)


def get_num_open_samples() -> int:
    """Return the number of samples in the shared open dataset."""
    y_path = DATA_ROOT / "open" / "y.npy"
    return len(np.load(y_path))
