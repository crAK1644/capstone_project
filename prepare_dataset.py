"""
Dataset preparation for SSFL (Semisupervised Federated Learning) intrusion detection.

Replicates the data pipeline from:
  Zhao et al., "Semisupervised Federated-Learning-Based Intrusion Detection
  Method for Internet of Things," IEEE IoT Journal, 2023.

Steps:
  1. Build mini-N-BaIoT: sample 1000 records per (device, traffic category).
  2. Split into private (70%), open (10%), test (20%) — disjoint.
  3. Min-max normalise features to [0, 1].
  4. Reshape each 115-d vector into a 23×5 matrix (5 time windows).
  5. Distribute private data to clients under 3 non-IID scenarios.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "prepared_data"

SEED = 42

# 11 traffic categories (label indices)
LABEL_MAP: dict[str, int] = {
    "benign": 0,
    "gafgyt.combo": 1,
    "gafgyt.junk": 2,
    "gafgyt.scan": 3,
    "gafgyt.tcp": 4,
    "gafgyt.udp": 5,
    "mirai.ack": 6,
    "mirai.scan": 7,
    "mirai.syn": 8,
    "mirai.udp": 9,
    "mirai.udpplain": 10,
}

LABEL_NAMES: list[str] = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

DEVICE_IDS = list(range(1, 10))  # 1..9

# Number of features and time-window structure
NUM_FEATURES = 115
NUM_ROWS = 23   # features per time window
NUM_COLS = 5    # number of time windows

SAMPLES_PER_CLASS = 1000   # mini-N-BaIoT: 1000 per (device, label)

# Split ratios
PRIVATE_RATIO = 0.70
OPEN_RATIO = 0.10
TEST_RATIO = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attack_type_from_filename(fname: str) -> str:
    """Extract attack type string from filename like '1.gafgyt.combo.csv'."""
    parts = fname.split(".")
    # e.g. ['1', 'gafgyt', 'combo', 'csv'] -> 'gafgyt.combo'
    # e.g. ['1', 'benign', 'csv'] -> 'benign'
    return ".".join(parts[1:-1])  # drop device id prefix and .csv suffix


def _get_device_files(device_id: int) -> list[tuple[str, Path]]:
    """Return list of (attack_type, filepath) for a device."""
    files = []
    for f in sorted(DATA_DIR.glob(f"{device_id}.*.csv")):
        attack = _attack_type_from_filename(f.name)
        if attack in LABEL_MAP:
            files.append((attack, f))
    return files


# ---------------------------------------------------------------------------
# Step 1: Build mini-N-BaIoT
# ---------------------------------------------------------------------------

def build_mini_nbaiot(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample 1000 records per (device, traffic category).

    Returns
    -------
    X : ndarray, shape (N, 115)
    y : ndarray, shape (N,)        — integer labels 0..10
    device_ids : ndarray, shape (N,) — device id 1..9
    """
    xs, ys, ds = [], [], []

    for dev_id in DEVICE_IDS:
        for attack, fpath in _get_device_files(dev_id):
            label = LABEL_MAP[attack]
            df = pd.read_csv(fpath, header=0)
            n = len(df)
            k = min(SAMPLES_PER_CLASS, n)
            idx = rng.choice(n, size=k, replace=False)
            xs.append(df.iloc[idx].values.astype(np.float32))
            ys.append(np.full(k, label, dtype=np.int64))
            ds.append(np.full(k, dev_id, dtype=np.int64))
            print(f"  Device {dev_id} | {attack:20s} | sampled {k}/{n}")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    device_ids = np.concatenate(ds, axis=0)
    print(f"\nmini-N-BaIoT total: {len(X)} samples "
          f"({len(np.unique(y))} classes, {len(np.unique(device_ids))} devices)\n")
    return X, y, device_ids


# ---------------------------------------------------------------------------
# Step 2: Split into private / open / test (per device per class)
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    device_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Stratified split, preserving device and label proportions.

    Returns dict with keys 'private', 'open', 'test', each mapping to
    (X, y, device_ids).
    """
    splits: dict[str, list] = {"private": [[], [], []], "open": [[], [], []], "test": [[], [], []]}

    for dev_id in np.unique(device_ids):
        for label in np.unique(y[device_ids == dev_id]):
            mask = (device_ids == dev_id) & (y == label)
            idx = np.where(mask)[0]
            rng.shuffle(idx)

            n = len(idx)
            n_priv = int(n * PRIVATE_RATIO)
            n_open = int(n * OPEN_RATIO)

            priv_idx = idx[:n_priv]
            open_idx = idx[n_priv:n_priv + n_open]
            test_idx = idx[n_priv + n_open:]

            splits["private"][0].append(X[priv_idx])
            splits["private"][1].append(y[priv_idx])
            splits["private"][2].append(device_ids[priv_idx])

            splits["open"][0].append(X[open_idx])
            splits["open"][1].append(y[open_idx])
            splits["open"][2].append(device_ids[open_idx])

            splits["test"][0].append(X[test_idx])
            splits["test"][1].append(y[test_idx])
            splits["test"][2].append(device_ids[test_idx])

    result = {}
    for name in splits:
        result[name] = (
            np.concatenate(splits[name][0]),
            np.concatenate(splits[name][1]),
            np.concatenate(splits[name][2]),
        )
    for name, (xd, yd, dd) in result.items():
        print(f"  {name:8s}: {len(xd):6d} samples")
    print()
    return result


# ---------------------------------------------------------------------------
# Step 3: Min-max normalisation
# ---------------------------------------------------------------------------

def normalise(
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Fit min-max on private set, apply to all splits.  Returns updated data
    dict plus (feat_min, feat_max) arrays.
    """
    X_priv = data["private"][0]
    feat_min = X_priv.min(axis=0)
    feat_max = X_priv.max(axis=0)

    # Avoid division by zero for constant features
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0

    normed = {}
    for name, (X, y, d) in data.items():
        X_norm = (X - feat_min) / denom
        X_norm = np.clip(X_norm, 0.0, 1.0)
        normed[name] = (X_norm.astype(np.float32), y, d)

    print("  Min-max normalisation applied (fit on private set).\n")
    return normed, feat_min, feat_max


# ---------------------------------------------------------------------------
# Step 4: Reshape 115-d vector → 23×5 matrix
# ---------------------------------------------------------------------------

def reshape_2d(X: np.ndarray) -> np.ndarray:
    """Reshape (N, 115) → (N, 1, 23, 5) for CNN input (1 channel)."""
    N = X.shape[0]
    # Split into 5 groups of 23 features (consecutive) → columns
    X_2d = X.reshape(N, NUM_COLS, NUM_ROWS).transpose(0, 2, 1)  # (N, 23, 5)
    return X_2d[:, np.newaxis, :, :]  # (N, 1, 23, 5)


# ---------------------------------------------------------------------------
# Step 5: Non-IID client distributions
# ---------------------------------------------------------------------------

@dataclass
class ClientData:
    client_id: int
    device_id: int
    X: np.ndarray
    y: np.ndarray
    labels_present: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.labels_present = sorted(np.unique(self.y).tolist())


def _shard_split(
    X: np.ndarray,
    y: np.ndarray,
    device_id: int,
    K_di: int,
    rng: np.random.Generator,
    client_id_offset: int,
) -> list[ClientData]:
    """
    Scenario 1/2 shard strategy:
    Sort by label, divide into 2*K_di shards of equal size,
    assign 2 shards per client.
    """
    # Sort by label
    order = np.argsort(y, kind="stable")
    X_sorted = X[order]
    y_sorted = y[order]

    n = len(X_sorted)
    num_shards = 2 * K_di
    shard_size = n // num_shards

    # Create shards
    shards_X = []
    shards_y = []
    for s in range(num_shards):
        start = s * shard_size
        end = start + shard_size if s < num_shards - 1 else n
        shards_X.append(X_sorted[start:end])
        shards_y.append(y_sorted[start:end])

    # Shuffle shard indices and assign 2 per client
    shard_indices = np.arange(num_shards)
    rng.shuffle(shard_indices)

    clients = []
    for k in range(K_di):
        s1, s2 = shard_indices[2 * k], shard_indices[2 * k + 1]
        cx = np.concatenate([shards_X[s1], shards_X[s2]])
        cy = np.concatenate([shards_y[s1], shards_y[s2]])
        clients.append(ClientData(
            client_id=client_id_offset + k,
            device_id=device_id,
            X=cx,
            y=cy,
        ))
    return clients


def _dirichlet_split(
    X: np.ndarray,
    y: np.ndarray,
    device_id: int,
    K_di: int,
    alpha: float,
    rng: np.random.Generator,
    client_id_offset: int,
) -> list[ClientData]:
    """
    Scenario 3: Dirichlet distribution split.
    """
    labels = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(K_di)]

    for label in labels:
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        proportions = rng.dirichlet(np.full(K_di, alpha))
        # Compute number per client
        counts = (proportions * len(label_idx)).astype(int)
        # Distribute remainder
        remainder = len(label_idx) - counts.sum()
        for i in range(remainder):
            counts[i % K_di] += 1

        offset = 0
        for k in range(K_di):
            client_indices[k].extend(label_idx[offset:offset + counts[k]].tolist())
            offset += counts[k]

    clients = []
    for k in range(K_di):
        idx = np.array(client_indices[k])
        if len(idx) == 0:
            continue
        clients.append(ClientData(
            client_id=client_id_offset + k,
            device_id=device_id,
            X=X[idx],
            y=y[idx],
        ))
    return clients


def _num_classes_for_device(device_id: int, y_device: np.ndarray) -> int:
    """Return L_di — number of distinct labels available for this device."""
    return len(np.unique(y_device))


def build_scenario(
    X_priv: np.ndarray,
    y_priv: np.ndarray,
    d_priv: np.ndarray,
    scenario: int,
    rng: np.random.Generator,
) -> list[ClientData]:
    """
    Build client splits for a given scenario (1, 2, or 3).

    Scenario 1: K_di = 3 per device, shard strategy
    Scenario 2: K_di = L_di per device, shard strategy
    Scenario 3: K_di = L_di per device, Dirichlet(α=0.1)
    """
    all_clients: list[ClientData] = []
    client_id_offset = 0

    for dev_id in np.unique(d_priv):
        mask = d_priv == dev_id
        X_dev = X_priv[mask]
        y_dev = y_priv[mask]
        L_di = _num_classes_for_device(dev_id, y_dev)

        if scenario == 1:
            K_di = 3
            clients = _shard_split(X_dev, y_dev, dev_id, K_di, rng, client_id_offset)
        elif scenario == 2:
            K_di = L_di
            clients = _shard_split(X_dev, y_dev, dev_id, K_di, rng, client_id_offset)
        elif scenario == 3:
            K_di = L_di
            clients = _dirichlet_split(X_dev, y_dev, dev_id, K_di, 0.1, rng, client_id_offset)
        else:
            raise ValueError(f"Unknown scenario {scenario}")

        all_clients.extend(clients)
        client_id_offset += len(clients)

    return all_clients


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_dataset(output_dir: Path, data: dict, scenarios: dict, feat_min, feat_max):
    """Save all prepared artefacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    for name in ("private", "open", "test"):
        X, y, d = data[name]
        split_dir = output_dir / name
        split_dir.mkdir(exist_ok=True)

        # Save as flat arrays
        np.save(split_dir / "X.npy", X)
        np.save(split_dir / "y.npy", y)
        np.save(split_dir / "device_ids.npy", d)

        # Also save reshaped version for CNN
        X_2d = reshape_2d(X)
        np.save(split_dir / "X_2d.npy", X_2d)

        print(f"  Saved {name}: X{X.shape}, X_2d{X_2d.shape}, y{y.shape}")

    # Save normalisation params
    np.save(output_dir / "feat_min.npy", feat_min)
    np.save(output_dir / "feat_max.npy", feat_max)

    # Save scenario client distributions
    for scenario_num, clients in scenarios.items():
        sc_dir = output_dir / f"scenario_{scenario_num}"
        sc_dir.mkdir(exist_ok=True)
        summary = []
        for c in clients:
            cdir = sc_dir / f"client_{c.client_id}"
            cdir.mkdir(exist_ok=True)
            np.save(cdir / "X.npy", c.X)
            np.save(cdir / "y.npy", c.y)

            # Also save 2D
            X_2d = reshape_2d(c.X)
            np.save(cdir / "X_2d.npy", X_2d)

            summary.append({
                "client_id": int(c.client_id),
                "device_id": int(c.device_id),
                "num_samples": int(len(c.y)),
                "labels_present": [int(l) for l in c.labels_present],
            })

        with open(sc_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Scenario {scenario_num}: {len(clients)} clients saved")

    # Save label map
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"\nAll data saved to {output_dir}")


def load_split(split_name: str, use_2d: bool = True, base_dir: Path = OUTPUT_DIR):
    """Load a saved split as PyTorch tensors."""
    split_dir = base_dir / split_name
    X_key = "X_2d.npy" if use_2d else "X.npy"
    X = torch.from_numpy(np.load(split_dir / X_key))
    y = torch.from_numpy(np.load(split_dir / "y.npy"))
    return X, y


def load_client(
    scenario: int,
    client_id: int,
    use_2d: bool = True,
    base_dir: Path = OUTPUT_DIR,
):
    """Load a single client's data for a scenario as PyTorch tensors."""
    cdir = base_dir / f"scenario_{scenario}" / f"client_{client_id}"
    X_key = "X_2d.npy" if use_2d else "X.npy"
    X = torch.from_numpy(np.load(cdir / X_key))
    y = torch.from_numpy(np.load(cdir / "y.npy"))
    return X, y


def load_scenario_summary(scenario: int, base_dir: Path = OUTPUT_DIR) -> list[dict]:
    """Load client summary for a scenario."""
    with open(base_dir / f"scenario_{scenario}" / "summary.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("Step 1: Building mini-N-BaIoT dataset")
    print("=" * 60)
    X, y, device_ids = build_mini_nbaiot(rng)

    print("=" * 60)
    print("Step 2: Splitting into private / open / test (70/10/20)")
    print("=" * 60)
    data = split_data(X, y, device_ids, rng)

    print("=" * 60)
    print("Step 3: Min-max normalisation")
    print("=" * 60)
    data, feat_min, feat_max = normalise(data)

    print("=" * 60)
    print("Step 4 & 5: Building non-IID client scenarios")
    print("=" * 60)
    scenarios = {}
    for sc in (1, 2, 3):
        print(f"\n--- Scenario {sc} ---")
        # Use a fresh rng copy so scenarios are independent
        sc_rng = np.random.default_rng(SEED + sc)
        clients = build_scenario(
            *data["private"], scenario=sc, rng=sc_rng,
        )
        scenarios[sc] = clients
        total_clients = len(clients)
        total_samples = sum(len(c.y) for c in clients)
        print(f"  {total_clients} clients, {total_samples} total private samples")

        # Print distribution summary
        for c in clients:
            labels_str = ",".join(str(l) for l in c.labels_present)
            print(f"    Client {c.client_id:3d} (dev {c.device_id}): "
                  f"{len(c.y):5d} samples, labels=[{labels_str}]")

    print("\n" + "=" * 60)
    print("Saving all artefacts")
    print("=" * 60)
    save_dataset(OUTPUT_DIR, data, scenarios, feat_min, feat_max)

    # Print final summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for name in ("private", "open", "test"):
        X_s, y_s, d_s = data[name]
        print(f"\n{name.upper()}: {len(y_s)} samples")
        for label_idx in range(len(LABEL_NAMES)):
            count = (y_s == label_idx).sum()
            if count > 0:
                print(f"  {LABEL_NAMES[label_idx]:20s} (label {label_idx:2d}): {count:5d}")

    print("\nScenario summary:")
    for sc, clients in scenarios.items():
        print(f"  Scenario {sc}: {len(clients)} clients")

    print("\nDone! Use load_split() and load_client() to access the data.")


if __name__ == "__main__":
    main()
