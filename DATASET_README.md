# SSFL Dataset Preparation

Dataset pipeline for reproducing the experiments from:

> Zhao et al., *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things,"*
> IEEE Internet of Things Journal, Vol. 10, No. 10, May 2023.

## Overview

This pipeline transforms the raw **N-BaIoT** dataset (9 IoT devices, 11 traffic categories, 115 features) into the **mini-N-BaIoT** format described in the paper, ready for federated learning experiments.

## Dataset Summary

| Property | Value |
|---|---|
| Source dataset | N-BaIoT (UCI ML Repository) |
| Devices | 9 IoT devices (doorbells, cameras, thermostats, etc.) |
| Traffic classes | 11 (1 benign + 5 gafgyt attacks + 5 mirai attacks) |
| Features per sample | 115 (from 5 time windows × 23 features) |
| Samples per (device, class) | 1,000 |
| Total mini-N-BaIoT samples | 89,000 |

### Traffic Classes (Label Map)

| Label | Category | Type |
|---|---|---|
| 0 | benign | Normal |
| 1 | gafgyt.combo | Attack |
| 2 | gafgyt.junk | Attack |
| 3 | gafgyt.scan | Attack |
| 4 | gafgyt.tcp | Attack |
| 5 | gafgyt.udp | Attack |
| 6 | mirai.ack | Attack |
| 7 | mirai.scan | Attack |
| 8 | mirai.syn | Attack |
| 9 | mirai.udp | Attack |
| 10 | mirai.udpplain | Attack |

> **Note:** Devices 3 and 7 only have 6 classes (benign + 5 gafgyt). The others have all 11.

### Devices

| ID | Device |
|---|---|
| 1 | Danmini Doorbell |
| 2 | Ecobee Thermostat |
| 3 | Ennio Doorbell |
| 4 | Philips B120N10 Baby Monitor |
| 5 | Provision PT 737E Security Camera |
| 6 | Provision PT 838 Security Camera |
| 7 | Samsung SNH 1011 N Webcam |
| 8 | SimpleHome XCS7 1002 WHT Security Camera |
| 9 | SimpleHome XCS7 1003 WHT Security Camera |

## Pipeline Steps

### Step 1 — Build mini-N-BaIoT

Randomly samples **1,000 records** from each `(device, traffic category)` pair, producing 89,000 total samples.

### Step 2 — Train / Open / Test Split (70% / 10% / 20%)

Stratified by device and label. The three sets are **disjoint**.

| Split | Samples | Purpose |
|---|---|---|
| **Private** | 62,300 | Distributed to FL clients for local training |
| **Open** | 8,900 | Unlabeled data shared across all clients (for distillation) |
| **Test** | 17,800 | Held out for evaluation |

### Step 3 — Min-Max Normalisation

All features scaled to `[0, 1]`. Fitted on the private set only (to avoid data leakage), then applied to all splits.

### Step 4 — 2D Reshaping

Each 115-d feature vector is reshaped into a **23 × 5 matrix** (23 features per time window, 5 time windows), stored as `(N, 1, 23, 5)` tensors (1 channel for CNN input).

### Step 5 — Non-IID Client Scenarios

Private data is distributed to clients under three scenarios:

| Scenario | Clients | Strategy |
|---|---|---|
| **1** | 27 | K=3 per device, shard-based (sort by label, 2 shards per client) |
| **2** | 89 | K=L per device, shard-based (L = number of classes for that device) |
| **3** | 89 | K=L per device, Dirichlet(α=0.1) distribution |

## How to Run

### Prerequisites

```bash
# Install dependencies (if not already done)
uv add numpy pandas torch
```

### Generate the Dataset

```bash
uv run python prepare_dataset.py
```

This creates the `prepared_data/` directory with all artefacts.

### Output Structure

```
prepared_data/
├── label_map.json          # Label name → index mapping
├── feat_min.npy            # Min values for normalisation
├── feat_max.npy            # Max values for normalisation
├── private/
│   ├── X.npy               # (62300, 115) flat features
│   ├── X_2d.npy            # (62300, 1, 23, 5) CNN-ready
│   ├── y.npy               # (62300,) integer labels
│   └── device_ids.npy      # (62300,) device IDs
├── open/
│   ├── X.npy               # (8900, 115)
│   ├── X_2d.npy            # (8900, 1, 23, 5)
│   ├── y.npy               # (8900,) — labels exist but NOT used during training
│   └── device_ids.npy
├── test/
│   ├── X.npy               # (17800, 115)
│   ├── X_2d.npy            # (17800, 1, 23, 5)
│   ├── y.npy               # (17800,)
│   └── device_ids.npy
├── scenario_1/
│   ├── summary.json        # Client metadata (ID, device, sample count, labels)
│   ├── client_0/
│   │   ├── X.npy           # Flat features for this client
│   │   ├── X_2d.npy        # CNN-ready features
│   │   └── y.npy           # Labels
│   ├── client_1/
│   │   └── ...
│   └── ...                 # 27 clients total
├── scenario_2/
│   └── ...                 # 89 clients
└── scenario_3/
    └── ...                 # 89 clients
```

## How to Load the Data

### In Python (PyTorch)

```python
from prepare_dataset import load_split, load_client, load_scenario_summary

# --- Load global splits ---
X_test, y_test = load_split("test", use_2d=True)
# X_test shape: (17800, 1, 23, 5), dtype: float32
# y_test shape: (17800,), dtype: int64

X_open, y_open = load_split("open", use_2d=True)
# Open set — y_open exists but is NOT used during SSFL training

X_priv, y_priv = load_split("private", use_2d=True)

# --- Load flat features instead ---
X_test_flat, y_test_flat = load_split("test", use_2d=False)
# X_test_flat shape: (17800, 115)

# --- Load a client's data for a specific scenario ---
X_client, y_client = load_client(scenario=1, client_id=0)
# X_client shape: (2568, 1, 23, 5)

# --- Get scenario metadata ---
summary = load_scenario_summary(scenario=1)
# Returns list of dicts with: client_id, device_id, num_samples, labels_present
for client in summary:
    print(f"Client {client['client_id']}: {client['num_samples']} samples, "
          f"labels={client['labels_present']}")
```

### Using Raw NumPy Files

```python
import numpy as np

X = np.load("prepared_data/test/X_2d.npy")   # (17800, 1, 23, 5)
y = np.load("prepared_data/test/y.npy")       # (17800,)
```

### Creating a PyTorch DataLoader

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from prepare_dataset import load_split, load_client

# Test set loader
X_test, y_test = load_split("test")
test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

# Client loader for federated training
X_c, y_c = load_client(scenario=1, client_id=0)
client_ds = TensorDataset(X_c, y_c)
client_loader = DataLoader(client_ds, batch_size=100, shuffle=True)
```

## Reproducibility

- Random seed is fixed at `42` for mini-N-BaIoT sampling and data splitting.
- Each scenario uses seed `42 + scenario_number` for independent client distributions.
- Re-running `prepare_dataset.py` produces identical output.

## Reference

```bibtex
@article{zhao2023ssfl,
  title={Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things},
  author={Zhao, Ruijie and Wang, Yijun and Xue, Zhi and Ohtsuki, Tomoaki and Adebisi, Bamidele and Gui, Guan},
  journal={IEEE Internet of Things Journal},
  volume={10},
  number={10},
  pages={8645--8657},
  year={2023}
}
```
