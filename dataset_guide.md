# Dataset Comparison & Flower Integration Guide

> **Project:** Semisupervised Federated Learning (SSFL) for IoT Intrusion Detection  
> **Paper:** Zhao et al., *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things"*, IEEE IoT Journal, Vol. 10, No. 10, May 2023.

---

## Table of Contents

1. [Source: The N-BaIoT Dataset](#1-source-the-n-baiot-dataset)
2. [Dataset Comparison: `prepared_data` vs `prepared_data_ml`](#2-dataset-comparison-prepared_data-vs-prepared_data_ml)
   - [Overview Table](#21-overview-table)
   - [File Structure Side-by-Side](#22-file-structure-side-by-side)
   - [Array Shapes and Types](#23-array-shapes-and-types)
   - [Federated Scenarios](#24-federated-scenarios)
   - [Disk Usage](#25-disk-usage)
   - [When to Use Which](#26-when-to-use-which)
3. [Non-IID Scenarios In Depth](#3-non-iid-scenarios-in-depth)
4. [SSFL Algorithm — Data Flow Summary](#4-ssfl-algorithm--data-flow-summary)
5. [Flower Integration Guide](#5-flower-integration-guide)
   - [Flower Architecture Overview](#51-flower-architecture-overview)
   - [Loading Data in a Flower Client](#52-loading-data-in-a-flower-client)
   - [Client App (`client_app.py`)](#53-client-app-client_apppy)
   - [Server App & Custom Strategy (`server_app.py`)](#54-server-app--custom-strategy-server_apppy)
   - [Which Dataset to Use Where in Flower](#55-which-dataset-to-use-where-in-flower)
6. [Quick Reference: Loading Snippets](#6-quick-reference-loading-snippets)

---

## 1. Source: The N-BaIoT Dataset

Both prepared datasets derive from the same raw source located in `data/`.

| Property | Value |
|---|---|
| Origin | UCI ML Repository — "Detection of IoT Botnet Attacks N-BaIoT" |
| Devices | 9 commercial IoT devices (doorbells, cameras, thermostat, webcam) |
| Botnets | Mirai and BASHLITE (Gafgyt) |
| Raw samples | ~7,062,606 total rows across all CSVs |
| Features | 115 per sample (statistical aggregates over 5 time-decay windows) |
| File format | One CSV per `(device_id, traffic_class)` pair — 89 files total |

### Traffic Classes (11 total)

| Label | Class Name | Type |
|---|---|---|
| 0 | `benign` | Normal |
| 1 | `gafgyt.combo` | Attack |
| 2 | `gafgyt.junk` | Attack |
| 3 | `gafgyt.scan` | Attack |
| 4 | `gafgyt.tcp` | Attack |
| 5 | `gafgyt.udp` | Attack |
| 6 | `mirai.ack` | Attack (7 devices only) |
| 7 | `mirai.scan` | Attack (7 devices only) |
| 8 | `mirai.syn` | Attack (7 devices only) |
| 9 | `mirai.udp` | Attack (7 devices only) |
| 10 | `mirai.udpplain` | Attack (7 devices only) |

> **Note:** Devices 3 (Ennio Doorbell) and 7 (Samsung Webcam) were not infected by Mirai — they only have 6 classes (labels 0–5).

### Feature Structure (115 Features = 5 Windows × 23 Stats)

Each 115-d vector is computed by the Kitsune framework from 4 stream aggregation types:

| Stream | Description |
|---|---|
| `H` | Recent traffic from this packet's source host |
| `HH` | Recent traffic from source host → destination host |
| `HH_jit` | Jitter of the source → destination stream |
| `HpHp` | Source host+port → destination host+port |

Each stream is summarized over 5 time-decay windows (λ = 5, 3, 1, 0.1, 0.01) using statistics: `weight`, `mean`, `variance`, `radius`, `magnitude`, `covariance`, `pcc`.

---

## 2. Dataset Comparison: `prepared_data` vs `prepared_data_ml`

### 2.1 Overview Table

| Property | `prepared_data` | `prepared_data_ml` |
|---|---|---|
| **Purpose** | CNN-based SSFL (PyTorch Conv2D) | General ML / tabular models |
| **Source** | `data/` raw CSVs | `data/` raw CSVs (same pipeline) |
| **Total samples** | 89,000 | 89,000 |
| **Feature format** | Flat `X.npy` (N, 115) + CNN tensor `X_2d.npy` (N, 1, 23, 5) | Flat `X.npy` (N, 115) only |
| **Normalization** | Min-Max [0,1], fit on private set | Min-Max [0,1], fit on private set |
| **Splits** | private / open / test | private / open / test |
| **Scenarios** | 3 (same client distributions) | 3 (same client distributions) |
| **Total files** | 633 | 425 |
| **Disk usage** | ~257 MB | ~130 MB |
| **dtype (X)** | float32 | float32 |
| **dtype (y)** | int64 | int64 |
| **CNN-ready** | Yes — `X_2d.npy` is `(N, 1, 23, 5)` | No — only flat 2D tabular |
| **sklearn-ready** | Yes — `X.npy` is `(N, 115)` | Yes — `X.npy` is `(N, 115)` |
| **Seed** | 42 | 42 |

> Both datasets use **identical** random seeds, sampling strategy, split ratios, and normalization statistics — meaning the same samples end up in the same splits. The only structural difference is the presence or absence of `X_2d.npy`.

---

### 2.2 File Structure Side-by-Side

```
prepared_data/                      prepared_data_ml/
├── label_map.json                  ├── label_map.json
├── feat_min.npy   (115,)           ├── feat_min.npy   (115,)
├── feat_max.npy   (115,)           ├── feat_max.npy   (115,)
│                                   │
├── private/                        ├── private/
│   ├── X.npy        (62300, 115)   │   ├── X.npy        (62300, 115)
│   ├── X_2d.npy  (62300,1,23,5) ◄─┤   │   ← ABSENT in ml version
│   ├── y.npy        (62300,)       │   ├── y.npy        (62300,)
│   └── device_ids.npy (62300,)     │   └── device_ids.npy (62300,)
│                                   │
├── open/                           ├── open/
│   ├── X.npy         (8900, 115)   │   ├── X.npy         (8900, 115)
│   ├── X_2d.npy   (8900,1,23,5) ◄─┤   │   ← ABSENT in ml version
│   ├── y.npy         (8900,)       │   ├── y.npy         (8900,)  [NOT used in SSFL]
│   └── device_ids.npy  (8900,)     │   └── device_ids.npy  (8900,)
│                                   │
├── test/                           ├── test/
│   ├── X.npy        (17800, 115)   │   ├── X.npy        (17800, 115)
│   ├── X_2d.npy  (17800,1,23,5) ◄─┤   │   ← ABSENT in ml version
│   ├── y.npy        (17800,)       │   ├── y.npy        (17800,)
│   └── device_ids.npy (17800,)     │   └── device_ids.npy (17800,)
│                                   │
├── scenario_1/                     ├── scenario_1/
│   ├── summary.json  (27 clients)  │   ├── summary.json  (27 clients)
│   └── client_0/                   │   └── client_0/
│       ├── X.npy                   │       ├── X.npy
│       ├── X_2d.npy  ◄─────────────┤       │   ← ABSENT in ml version
│       └── y.npy                   │       └── y.npy
│                                   │
├── scenario_2/  (89 clients)       ├── scenario_2/  (89 clients)
└── scenario_3/  (89 clients)       └── scenario_3/  (89 clients)
```

---

### 2.3 Array Shapes and Types

| File | `prepared_data` shape | `prepared_data_ml` shape | Notes |
|---|---|---|---|
| `private/X.npy` | (62300, 115) float32 | (62300, 115) float32 | Identical |
| `private/X_2d.npy` | (62300, 1, 23, 5) float32 | **Does not exist** | CNN-only |
| `private/y.npy` | (62300,) int64 | (62300,) int64 | Identical |
| `private/device_ids.npy` | (62300,) int64 | (62300,) int64 | Identical |
| `open/X.npy` | (8900, 115) float32 | (8900, 115) float32 | Identical |
| `open/X_2d.npy` | (8900, 1, 23, 5) float32 | **Does not exist** | CNN-only |
| `test/X.npy` | (17800, 115) float32 | (17800, 115) float32 | Identical |
| `test/X_2d.npy` | (17800, 1, 23, 5) float32 | **Does not exist** | CNN-only |
| `client_k/X.npy` | (N_k, 115) float32 | (N_k, 115) float32 | Identical |
| `client_k/X_2d.npy` | (N_k, 1, 23, 5) float32 | **Does not exist** | CNN-only |
| `client_k/y.npy` | (N_k,) int64 | (N_k,) int64 | Identical |
| `feat_min.npy` | (115,) float32 | (115,) float32 | Identical |
| `feat_max.npy` | (115,) float32 | (115,) float32 | Identical |

The `X_2d` reshape logic is: `X.reshape(N, 1, 23, 5)` — treating the 5 time-decay windows as spatial width and 23 per-window statistics as input channels. This is a semantically meaningful layout for 2D convolutions that capture cross-feature and cross-window interactions.

---

### 2.4 Federated Scenarios

Both datasets share identical scenario structures. The client assignments are deterministic — the same client in `prepared_data/scenario_1/client_5/` and `prepared_data_ml/scenario_1/client_5/` contain the same samples.

| Scenario | Clients | Strategy | Seed | Avg labels/client | Avg samples/client |
|---|---|---|---|---|---|
| **1** | 27 | Shard-based, K=3 per device | 43 | ~4.5 | ~2,307 |
| **2** | 89 | Shard-based, K=L per device | 44 | 2.0 | 700 |
| **3** | 89 | Dirichlet (α=0.1) | 45 | ~6.0 | ~700 |

**Scenario 1** — Each device's data is sorted by label and split into 2×3=6 shards. Each of the 3 clients for that device receives 2 contiguous shards. Clients see ~5 of 11 classes.

**Scenario 2** — K equals the number of classes L per device (6 or 11). Each client gets 2 shards, covering exactly ~2 classes. Maximum label exclusivity — most classes are completely absent from most clients.

**Scenario 3** — Label proportions for each class are sampled from Dirichlet(0.1). A concentration parameter α=0.1 is very small — it produces sparse distributions where most of the probability mass falls on 1–2 labels. This creates extreme heterogeneity with high variance in client sample counts (min=9, max=1974).

---

### 2.5 Disk Usage

| Dataset | Files | Size | Per-file breakdown |
|---|---|---|---|
| `prepared_data` | 633 | ~257 MB | X (115-d) + X_2d (CNN 4D) + y + device_ids per split/client |
| `prepared_data_ml` | 425 | ~130 MB | X (115-d) + y + device_ids per split/client |

The ~127 MB difference is entirely the `X_2d.npy` files (4× the element count of `X.npy` due to the 4D shape).

---

### 2.6 When to Use Which

| Model type | Use | Reason |
|---|---|---|
| PyTorch CNN (Conv2D) | `prepared_data` | Needs `X_2d.npy` of shape `(N, 1, 23, 5)` as input |
| PyTorch MLP / Linear | Either | Load `X.npy`; `X_2d` is not needed |
| scikit-learn (RF, SVM, XGBoost, LogReg, etc.) | `prepared_data_ml` | Natively expects 2D arrays `(N, features)` |
| Federated SSFL (paper reproduction) | `prepared_data` | CNN backbone requires the 2D spatial layout |
| Federated SSFL with tabular models | `prepared_data_ml` | Drop-in replacement when not using CNN |
| Baseline comparison experiments | `prepared_data_ml` | Faster to load, lighter, sklearn-compatible |

---

## 3. Non-IID Scenarios In Depth

### Why Non-IID is Central to This Project

In a real IoT deployment, each device generates only its own traffic. A doorbell captures doorbell traffic; a security camera captures camera traffic. When you train a federated model across such devices, each client's local dataset is **not** a representative sample of the global distribution — this is the non-IID (non-independent-and-identically-distributed) problem.

Non-IID data causes two concrete issues in knowledge distillation-based FL:
1. **Biased local predictions:** A client trained only on Mirai attacks will predict -1 (unfamiliar) for all Gafgyt traffic when evaluating open data.
2. **Voting collapse:** If most clients predict -1 for a class, the global label assigned by voting will be wrong, poisoning the distillation step.

The SSFL discriminator is the paper's solution: it explicitly identifies which open-set samples are "unfamiliar" to each client and excludes those predictions from voting.

### Scenario Statistics

```
Scenario 1 (27 clients):
  Distribution type : Shard-based (K=3 per device)
  Clients           : 27
  Min samples       : 1,400
  Max samples       : 2,568
  Mean samples      : 2,307
  Std samples       : 485
  Avg labels/client : ~4.5 / 11

Scenario 2 (89 clients):
  Distribution type : Shard-based (K=L per device, 2 shards/client)
  Clients           : 89
  Min samples       : 700  (uniform — exact shard sizes)
  Max samples       : 700
  Std samples       : 0
  Avg labels/client : 2.0 / 11  ← most extreme label exclusivity

Scenario 3 (89 clients):
  Distribution type : Dirichlet(α=0.1)
  Clients           : 89
  Min samples       : 9
  Max samples       : 1,974
  Mean samples      : ~700
  Std samples       : ~474
  Avg labels/client : ~6.0 / 11  ← high variance in both count and labels
```

### Client Composition Example (Scenario 1)

```
Client 0  | Device 1 (Danmini Doorbell)   | Labels: [0, 1, 5, 6, 7]    | 2,567 samples
Client 1  | Device 1                      | Labels: [1, 2, 3, 7, 8, 9] | 2,567 samples
Client 2  | Device 1                      | Labels: [3, 4, 5, 9, 10]   | 2,566 samples
Client 3  | Device 2 (Ecobee Thermostat)  | Labels: [0, 1, 5, 6, 7]    | 2,567 samples
...
Client 24 | Device 9 (SimpleHome XCS7)    | Labels: [3, 4, 5, 9, 10]   | 2,566 samples
```

Notice: no single client has all 11 classes. This means the first-stage classifier (trained only on private data) will predict incorrectly or with low confidence on classes it has never seen.

---

## 4. SSFL Algorithm — Data Flow Summary

Understanding which data splits are used at each step of the SSFL algorithm is essential for correctly wiring a Flower implementation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SSFL Communication Round t                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SERVER                          CLIENT k                               │
│  ──────                          ────────                               │
│                                                                         │
│  Broadcast global labels Pˢ ──► [receive Pˢ or empty on round 1]       │
│                                   │                                     │
│                         Step 5 (if Pˢ exists):                          │
│                         Distillation training on OPEN data              │
│                         using Pˢ as supervision signal                  │
│                         Data: open/X.npy  Labels: Pˢ (from server)     │
│                                   │                                     │
│                         Step 1:                                         │
│                         Train Classifier on PRIVATE data                │
│                         Data: scenario_k/client_i/X.npy + y.npy        │
│                                   │                                     │
│                         Step 1 (cont.):                                 │
│                         Run OPEN data through classifier                │
│                         → confidence score cⱼᵏ = max(softmax(output))  │
│                         Data: open/X.npy                                │
│                                   │                                     │
│                         Step 2:                                         │
│                         Build Dᵏᵈ:                                     │
│                           - Low confidence open samples → "unfamiliar"  │
│                           - All private samples → "familiar"            │
│                         Train Discriminator on Dᵏᵈ                     │
│                                   │                                     │
│                         Step 3:                                         │
│                         Run OPEN data through discriminator             │
│                         → mark unfamiliar samples as -1                 │
│                         → hard-label remaining samples                  │
│                         Upload hard labels array (N_open,) to server    │
│                                   │                                     │
│  ◄── Collect hard labels from all K clients                             │
│                                                                         │
│  Step 4 (Vote & Broadcast):                                             │
│    For each open sample j:                                              │
│      Collect all non-(-1) votes from K clients                          │
│      Majority vote → global label Pˢ[j]                                │
│    Broadcast Pˢ → next round                                            │
│                                                                         │
│  Evaluate on TEST set → accuracy, F1                                    │
│  Data: test/X.npy + test/y.npy                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Split Roles Summary

| Split | SSFL Role | Labels used? | Who accesses it? |
|---|---|---|---|
| `private` (per client) | Classifier training (Step 1) + Discriminator building (Step 2) | Yes — supervised | Each client only its own shard |
| `open` | Confidence scoring (Step 1) + Discriminator filtering (Step 3) + Distillation (Step 5) | No — unlabeled during training | All clients (same shared data) |
| `test` | Global evaluation after each round | Yes — evaluation only | Server (centralised) |

---

## 5. Flower Integration Guide

### 5.1 Flower Architecture Overview

Flower (flwr) structures a federated learning experiment into three components:

```
┌──────────────────────────────────────────────────────────────┐
│                     Flower Simulation                         │
│                                                              │
│  ┌─────────────────┐         ┌──────────────────────────┐   │
│  │   ServerApp      │         │      ClientApp            │   │
│  │                  │         │  (one instance per        │   │
│  │  SSFLStrategy    │◄───────►│   client_id)              │   │
│  │  - configure_fit │         │                           │   │
│  │  - aggregate_fit │         │  fit(parameters, config)  │   │
│  │  - aggregate_eval│         │  evaluate(params, config) │   │
│  └─────────────────┘         └──────────────────────────┘   │
│                                                              │
│  Communication: hard labels array (NOT model weights)        │
└──────────────────────────────────────────────────────────────┘
```

**Key design principle for SSFL:** Standard Flower strategies (FedAvg, FedProx) exchange **model parameters**. SSFL exchanges **hard-label predictions** on the shared open dataset. You must implement a custom `Strategy` subclass.

---

### 5.2 Loading Data in a Flower Client

Both datasets share the same loading pattern. The only difference is whether you load `X.npy` (tabular, works for both) or `X_2d.npy` (CNN-only, `prepared_data` only).

#### For `prepared_data` (CNN — PyTorch Conv2D models)

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_client_data_cnn(scenario: int, client_id: int, batch_size: int = 100):
    """Load a client's private data as a CNN-ready DataLoader."""
    base = f"prepared_data/scenario_{scenario}/client_{client_id}"

    # Shape: (N, 1, 23, 5) — required for Conv2d input
    X = torch.tensor(np.load(f"{base}/X_2d.npy"), dtype=torch.float32)
    y = torch.tensor(np.load(f"{base}/y.npy"),    dtype=torch.long)

    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def load_open_data_cnn(batch_size: int = 100):
    """Load the shared unlabeled open set (CNN format, labels hidden)."""
    # Shape: (8900, 1, 23, 5)
    X = torch.tensor(np.load("prepared_data/open/X_2d.npy"), dtype=torch.float32)
    # y exists in the file but is NOT passed to clients during training
    return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)


def load_test_data_cnn(batch_size: int = 100):
    """Load the global test set for server-side evaluation."""
    X = torch.tensor(np.load("prepared_data/test/X_2d.npy"), dtype=torch.float32)
    y = torch.tensor(np.load("prepared_data/test/y.npy"),    dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
```

#### For `prepared_data_ml` (Tabular — sklearn / MLP / XGBoost)

```python
import numpy as np

def load_client_data_ml(scenario: int, client_id: int):
    """Load a client's private data as flat NumPy arrays."""
    base = f"prepared_data_ml/scenario_{scenario}/client_{client_id}"

    # Shape: (N_k, 115) — standard 2D tabular format
    X = np.load(f"{base}/X.npy")  # float32
    y = np.load(f"{base}/y.npy")  # int64
    return X, y


def load_open_data_ml():
    """Load the shared open set (labels available but NOT used in SSFL training)."""
    X = np.load("prepared_data_ml/open/X.npy")  # (8900, 115)
    return X  # return only X — y is withheld by design


def load_test_data_ml():
    """Load the global test set for evaluation."""
    X = np.load("prepared_data_ml/test/X.npy")  # (17800, 115)
    y = np.load("prepared_data_ml/test/y.npy")  # (17800,)
    return X, y
```

---

### 5.3 Client App (`client_app.py`)

The Flower `ClientApp` wraps the client-side SSFL logic. Each call to `fit()` corresponds to **one full communication round** from the client's perspective (Steps 1–3 + Step 5 from previous round's labels).

#### Structure for `prepared_data` (CNN model)

```python
import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

class SSFLClient(fl.client.NumPyClient):
    """
    Flower client for CNN-based SSFL.
    Uses prepared_data/ with X_2d.npy tensors.
    """

    def __init__(self, client_id: int, scenario: int, ...):
        self.client_id  = client_id
        self.scenario   = scenario

        # Load this client's private data (CNN-ready)
        self.private_loader = load_client_data_cnn(scenario, client_id)

        # Load shared open data (unlabeled, CNN-ready)
        self.open_loader = load_open_data_cnn()
        self.open_X      = np.load(f"prepared_data/open/X_2d.npy")  # for indexing

        # Initialise models
        self.classifier    = Classifier(num_classes=11)   # CNN backbone + 11-class head
        self.discriminator = Discriminator(num_classes=2)  # same backbone + 2-class head

    def fit(self, parameters, config):
        """
        One SSFL round:
          - parameters[0]: global hard labels Pˢ (int64 array, shape (8900,))
                           or empty array on round 1
        """
        global_labels = parameters[0].astype(np.int64)
        server_round  = int(config["server_round"])

        # Step 5: Distillation (skip on round 1 — no labels yet)
        if server_round > 1 and len(global_labels) > 0:
            distillation_train(
                self.classifier, self.open_loader, global_labels, ...
            )

        # Step 1: Train classifier on private data
        train_classifier(self.classifier, self.private_loader, ...)

        # Step 1 (cont.): Compute confidence scores on open data
        predictions, confidences = compute_confidence_scores(
            self.classifier, self.open_loader, ...
        )

        # Step 2: Build Dᵏᵈ, train discriminator
        disc_loader = build_discriminator_dataset(
            confidences, self.open_X, private_X=..., theta=np.median(confidences)
        )
        train_discriminator(self.discriminator, disc_loader, ...)

        # Step 3: Filter predictions via discriminator → hard labels
        hard_labels = filter_and_predict(
            self.classifier, self.discriminator, self.open_loader, ...
        )  # shape (8900,), dtype int64, -1 = unfamiliar

        # Upload hard labels to server (not model weights)
        return [hard_labels.astype(np.float64)], len(self.private_loader.dataset), {}
```

#### Structure for `prepared_data_ml` (tabular model — e.g. sklearn RF or MLP)

```python
import flwr as fl
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SSFLClientML(fl.client.NumPyClient):
    """
    Flower client for tabular-model SSFL.
    Uses prepared_data_ml/ with flat X.npy arrays (N, 115).
    """

    def __init__(self, client_id: int, scenario: int):
        self.client_id = client_id

        # Load data as flat NumPy arrays
        self.X_priv, self.y_priv = load_client_data_ml(scenario, client_id)
        self.X_open               = load_open_data_ml()  # (8900, 115)

        # Initialise model (replace with any sklearn-compatible estimator)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._is_fitted = False

    def fit(self, parameters, config):
        global_labels = parameters[0].astype(np.int64)
        server_round  = int(config["server_round"])

        # Step 5: Distillation — retrain on open data with global labels
        if server_round > 1 and len(global_labels) > 0:
            valid_mask = global_labels >= 0
            if valid_mask.sum() > 0:
                # Combine private + labeled open data for distillation
                X_combined = np.vstack([self.X_priv, self.X_open[valid_mask]])
                y_combined = np.concatenate([self.y_priv, global_labels[valid_mask]])
                self.model.fit(X_combined, y_combined)
                self._is_fitted = True

        # Step 1: Train classifier on private data
        self.model.fit(self.X_priv, self.y_priv)
        self._is_fitted = True

        # Step 1 (cont.): Confidence = max class probability on open data
        proba       = self.model.predict_proba(self.X_open)     # (8900, n_classes)
        confidences = proba.max(axis=1)                          # (8900,)
        predictions = proba.argmax(axis=1).astype(np.int64)     # (8900,)

        # Step 2 (simplified — no discriminator network for tabular):
        # Use confidence threshold as the discriminator surrogate
        theta = np.median(confidences)

        # Step 3: Filter low-confidence predictions → -1 (unfamiliar)
        hard_labels = predictions.copy()
        hard_labels[confidences < theta] = -1

        return [hard_labels.astype(np.float64)], len(self.y_priv), {}
```

---

### 5.4 Server App & Custom Strategy (`server_app.py`)

The server-side logic is **identical** regardless of which dataset variant you use — it only ever sees hard-label arrays, never raw features.

```python
import flwr as fl
import numpy as np
from flwr.common import Parameters, Scalar
from flwr.server.strategy import Strategy
from typing import Dict, List, Optional, Tuple, Union


class SSFLStrategy(Strategy):
    """
    Custom Flower strategy implementing SSFL server logic.

    Communication protocol:
      Server → Client : int64 array (N_open,)  — global hard labels Pˢ
      Client → Server : int64 array (N_open,)  — local hard labels (-1 = unfamiliar)

    This replaces FedAvg entirely — NO model weights are exchanged.
    """

    def __init__(
        self,
        num_open_samples: int = 8900,
        num_classes:      int = 11,
        fraction_fit:     float = 1.0,
        min_fit_clients:  int = 2,
    ):
        self.num_open    = num_open_samples
        self.num_classes = num_classes
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients

        # Global labels start as all-unknown
        self.global_labels = np.full(num_open_samples, -1, dtype=np.int64)

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Send current global labels to all clients.
        On round 1, send an empty array (no labels yet).
        """
        clients = client_manager.sample(
            num_clients=max(
                self.min_fit_clients,
                int(client_manager.num_available() * self.fraction_fit)
            )
        )
        config = {"server_round": server_round}

        if server_round == 1:
            # No labels to send yet
            payload = [np.array([], dtype=np.float64)]
        else:
            payload = [self.global_labels.astype(np.float64)]

        parameters = fl.common.ndarrays_to_parameters(payload)
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        """
        Step 4 — Vote & Broadcast (Eq. 17 from paper).

        Collect hard labels from all clients, apply majority voting
        per open-set sample to produce global labels Pˢ.

        -1 means a client marked that sample as "unfamiliar" — excluded from vote.
        """
        if not results:
            return None, {}

        # Collect all client predictions: list of (N_open,) arrays
        all_preds = []
        for _, fit_res in results:
            labels = fl.common.parameters_to_ndarrays(fit_res.parameters)[0]
            all_preds.append(labels.astype(np.int64))

        # Stack: (K, N_open)
        pred_matrix = np.stack(all_preds, axis=0)

        # Majority vote per sample
        new_labels = np.full(self.num_open, -1, dtype=np.int64)
        for j in range(self.num_open):
            votes = pred_matrix[:, j]
            valid = votes[votes >= 0]
            if len(valid) > 0:
                counts       = np.bincount(valid, minlength=self.num_classes)
                new_labels[j] = int(np.argmax(counts))

        self.global_labels = new_labels
        num_labelled = int((new_labels >= 0).sum())

        # Return updated labels — clients receive these next round via configure_fit
        payload    = [self.global_labels.astype(np.float64)]
        parameters = fl.common.ndarrays_to_parameters(payload)

        metrics: Dict[str, Scalar] = {
            "num_labelled": num_labelled,
            "labelled_fraction": num_labelled / self.num_open,
        }
        return parameters, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Skip per-client evaluation — use centralised test set instead."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        """
        Centralised evaluation on the test set.
        Uses the test split from either prepared_data or prepared_data_ml.
        """
        X_test, y_test = load_test_data_ml()   # or load_test_data_cnn()
        # ... evaluate your model on X_test, y_test
        return loss, {"accuracy": acc, "f1": f1}

    def initialize_parameters(self, client_manager):
        """No initial model parameters — SSFL starts from scratch."""
        return fl.common.ndarrays_to_parameters(
            [np.full(self.num_open, -1, dtype=np.float64)]
        )
```

---

### 5.5 Which Dataset to Use Where in Flower

| Flower component | `prepared_data` | `prepared_data_ml` |
|---|---|---|
| `ClientApp.__init__` | Load `X_2d.npy` (CNN input) | Load `X.npy` (tabular input) |
| `ClientApp.fit` — classifier training | PyTorch Conv2D forward pass | sklearn `.fit()` or MLP |
| `ClientApp.fit` — open data scoring | `X_2d.npy` through CNN | `X.npy` through `.predict_proba()` |
| `ServerApp.aggregate_fit` | Hard labels array — dataset-agnostic | Hard labels array — dataset-agnostic |
| `ServerApp.evaluate` | `test/X_2d.npy` (CNN eval) | `test/X.npy` (sklearn eval) |
| Model architecture | `Classifier(CNNBackbone)` — 8 conv layers | RF / XGBoost / MLP — your choice |
| Discriminator | `Discriminator(CNNBackbone)` — 2-class | Confidence threshold (median) |

The server's `aggregate_fit` and voting logic are **completely agnostic** to the dataset format — it only ever sees integer arrays of shape `(N_open,)`.

---

## 6. Quick Reference: Loading Snippets

### Load normalization statistics

```python
import numpy as np

# Same file in both datasets
feat_min = np.load("prepared_data_ml/feat_min.npy")  # (115,)
feat_max = np.load("prepared_data_ml/feat_max.npy")  # (115,)

# Apply to new raw data (e.g., at inference time)
X_raw_norm = np.clip((X_raw - feat_min) / (feat_max - feat_min), 0.0, 1.0)
```

### Load label map

```python
import json

with open("prepared_data_ml/label_map.json") as f:
    label_map = json.load(f)  # {"benign": 0, "gafgyt.combo": 1, ...}

inv_label_map = {v: k for k, v in label_map.items()}
# inv_label_map[0] → "benign"
```

### Load scenario summary (client metadata)

```python
import json

with open("prepared_data_ml/scenario_1/summary.json") as f:
    summary = json.load(f)

for client in summary:
    print(f"Client {client['client_id']:3d} | "
          f"Device {client['device_id']} | "
          f"{client['num_samples']:4d} samples | "
          f"Labels: {client['labels_present']}")
```

### Load all clients for a scenario (batch loading)

```python
import numpy as np
from pathlib import Path

def load_all_clients(dataset: str, scenario: int):
    """Returns list of (X, y) tuples for every client."""
    base = Path(dataset) / f"scenario_{scenario}"
    summary = json.load(open(base / "summary.json"))
    return [
        (
            np.load(base / f"client_{c['client_id']}" / "X.npy"),
            np.load(base / f"client_{c['client_id']}" / "y.npy"),
        )
        for c in summary
    ]

clients = load_all_clients("prepared_data_ml", scenario=1)
print(f"Loaded {len(clients)} clients")
print(f"Client 0: X={clients[0][0].shape}, y={clients[0][1].shape}")
```

### Full Flower simulation entry point

```python
import flwr as fl
import json
import numpy as np
from pathlib import Path

# Choose dataset
DATASET = "prepared_data_ml"   # or "prepared_data"
SCENARIO = 1

summary = json.load(open(f"{DATASET}/scenario_{SCENARIO}/summary.json"))
NUM_CLIENTS = len(summary)

def client_fn(context):
    client_id = int(context.node_config["partition-id"])
    # For ML version:
    return SSFLClientML(client_id=client_id, scenario=SCENARIO)
    # For CNN version:
    # return SSFLClient(client_id=client_id, scenario=SCENARIO)

strategy = SSFLStrategy(
    num_open_samples=8900,
    num_classes=11,
    fraction_fit=1.0,
    min_fit_clients=2,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=200),
    strategy=strategy,
)
```

---

> **Reference:** Zhao R., Wang Y., Xue Z., Ohtsuki T., Adebisi B., Gui G. — "Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things", *IEEE Internet of Things Journal*, Vol. 10, No. 10, pp. 8645–8657, May 2023. DOI: 10.1109/JIOT.2022.3175918
