from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from prepare_dataset import load_client, load_scenario_summary, load_split


def get_client_ids(scenario: int, base_dir: Path | None = None) -> list[int]:
    summary = load_scenario_summary(scenario=scenario, base_dir=base_dir) if base_dir else load_scenario_summary(scenario=scenario)
    return [int(item["client_id"]) for item in summary]


def _split_dataset(
    dataset: TensorDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 < val_ratio < 1.0:
        return dataset, TensorDataset(dataset.tensors[0][:0], dataset.tensors[1][:0])

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds


def make_client_loaders(
    scenario: int,
    client_id: int,
    batch_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_2d: bool = True,
    base_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    X, y = load_client(scenario=scenario, client_id=client_id, use_2d=use_2d, base_dir=base_dir) if base_dir else load_client(scenario=scenario, client_id=client_id, use_2d=use_2d)
    dataset = TensorDataset(X.float(), y.long())

    train_ds, val_ds = _split_dataset(dataset, val_ratio=val_ratio, seed=seed + client_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_test_loader(
    batch_size: int,
    use_2d: bool = True,
    base_dir: Path | None = None,
) -> DataLoader:
    X, y = load_split(split_name="test", use_2d=use_2d, base_dir=base_dir) if base_dir else load_split(split_name="test", use_2d=use_2d)
    dataset = TensorDataset(X.float(), y.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_open_loader(
    batch_size: int,
    use_2d: bool = True,
    base_dir: Path | None = None,
) -> DataLoader:
    X, y = load_split(split_name="open", use_2d=use_2d, base_dir=base_dir) if base_dir else load_split(split_name="open", use_2d=use_2d)
    dataset = TensorDataset(X.float(), y.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
