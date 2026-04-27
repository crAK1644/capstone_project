# Semi-Supervised Federated Learning (SSFL) — Flower Infrastructure Plan
## Capstone Project: Federated Pseudo Labeling for Semi-Supervised Intrusion Detection
### Scenario 1 Implementation Plan

> **Paper Reference:** "Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things" (Zhao et al., IEEE IoT Journal, 2023)
> **Scope:** Scenario 1 only — 27 clients total (3 clients per IoT device × 9 devices), shard-based non-IID partitioning.

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [Scenario 1 — Data Setup Explained](#2-scenario-1--data-setup-explained)
3. [System Architecture & File Structure](#3-system-architecture--file-structure)
    - 3.1 Scope: Starting Infrastructure vs. CNN Body
4. [Module: `data_preparation.py`](#4-module-data_preparationpy)
5. [Module: `model.py`](#5-module-modelpy)
6. [Module: `client.py`](#6-module-clientpy)
7. [Module: `strategy.py`](#7-module-strategypy)
8. [Module: `server.py`](#8-module-serverpy)
9. [Module: `main.py`](#9-module-mainpy)
10. [Module: `utils.py`](#10-module-utilspy)
    - 10A. Module: `config.py`
    - 10B. Module: `launch.py` (optional)
    - 10C. Run Modes: Separate Processes vs. Flower Simulation
11. [Key Data Structures Reference](#11-key-data-structures-reference)
12. [Full Training Round Flow](#12-full-training-round-flow)
13. [Flower Communication Contract](#13-flower-communication-contract)
14. [Metrics & Evaluation Protocol](#14-metrics--evaluation-protocol)

---

## 1. Conceptual Overview

### What the System Does

The SSFL system trains an intrusion detection model across **27 simulated IoT clients** using the Flower federated learning framework. No raw data or model parameters are ever shared between clients and the server. Instead, clients share only **hard labels** (single integers representing a predicted traffic class) on a shared unlabeled open dataset.

Each training round consists of five sequential steps:

| Step | Who Executes | What Happens |
|------|-------------|--------------|
| **1. Train Classifier** | Client (local) | CNN trained on private labeled data using cross-entropy loss |
| **2. Train Discriminator** | Client (local) | CNN trained to distinguish "familiar" vs "unfamiliar" open samples |
| **3. Filter & Upload** | Client → Server | Discriminator filters predictions; client sends hard labels to server |
| **4. Vote & Broadcast** | Server | Majority vote across all clients determines global label per open sample |
| **5. Distillation** | Client (local) | Classifier fine-tuned on open data using global labels as teacher signal |

### Why Flower?

Flower (`flwr`) provides the communication backbone — it handles the network connections, round orchestration, and result collection between the server and all clients. We **override** its default aggregation logic (which normally averages model weights) to instead run our **voting mechanism** over hard labels.

### Key Design Decision: Parameters as Label Carriers

In standard Flower, `get_parameters()` and `set_parameters()` carry model weights. In our system, we repurpose these to carry:
- **Client → Server:** A numpy array of shape `(N_open,)` containing hard label predictions (integers 0–10, or -1 for "unfamiliar")
- **Server → Client:** A numpy array of shape `(N_open,)` containing global voted labels after the voting step

This means the `fit()` method on the client performs **all 5 SSFL steps** rather than just local training.

---

## 2. Scenario 1 — Data Setup Explained

### Dataset: N-BaIoT

The N-BaIoT dataset contains traffic from 9 IoT devices. Each device has up to 11 traffic categories (1 benign + up to 10 attack types). Each traffic sample has 115 features extracted from 5 time windows (100ms, 500ms, 1.5s, 10s, 1min).

**Important note on class counts:** Not every device carries all 11 classes. In particular, `Ennio_Doorbell` and `Samsung_SNH_1011_N_Webcam` are missing the Mirai attack family and therefore only carry 6 classes each. The rest of this document assumes a **global label space of size 11** (the union of all classes observed across the whole dataset): local class indices are mapped into the global 0–10 label space during data preparation, and any device that never emits a given class simply never uploads a vote for it. All server-side data structures (voting buckets, global label array) are sized to 11.

### Mini-N-BaIoT Construction

Before any partitioning, a mini version of the dataset is created:
- From each device `d_i`, take each traffic class `l` (up to 11 per device)
- Sample exactly **1000 records** from each `(device, class)` pair
- This gives at most 9,000 records per device (9 devices × up to 11 classes × 1000 samples)

### Dataset Splits

The combined mini dataset is split **globally** across all 9 devices:

| Split | Proportion | Has Labels | Purpose |
|-------|-----------|-----------|---------|
| Private (`D_p`) | 70% | Yes | Distributed to clients for local training |
| Open (`D_o`) | 10% | No (stripped) | Shared by all clients; used for distillation |
| Test (`D_test`) | 20% | Yes | Held out on server for evaluation |

The three sets are **disjoint** — no sample appears in more than one split.

### Scenario 1 Partitioning (Shard-Based)

For each device `d_i` with `K_di = 3` clients:

1. Take `D_p` for device `d_i` (the private labeled portion)
2. Sort all samples in `D_p` by their class label (ascending)
3. Divide the sorted data into `2 × K_di = 6` shards of equal size `|D_p| / 6`
4. Assign **2 consecutive shards** to each of the 3 clients

This results in each client having samples from **at most 2 adjacent class labels**, simulating highly non-IID distribution. With 9 devices × 3 clients each = **27 clients total**.

---

## 3. System Architecture & File Structure

```
ssfl_project/
├── data_preparation.py      # Dataset loading, normalization, partitioning (Scenario 1)
├── model.py                 # CNN architecture for both classifier and discriminator
├── client.py                # Flower NumPyClient — runs all 5 SSFL steps locally
├── strategy.py              # Custom Flower Strategy — voting mechanism on server
├── server.py                # Flower server startup and configuration
├── main.py                  # Entry point — launches server or client process
├── utils.py                 # Shared helper functions, logging, metrics
├── config.py                # All hyperparameters and constants in one place
└── data/
    ├── raw/                 # Raw N-BaIoT CSV files, one folder per device
    ├── processed/           # Saved mini-N-BaIoT after preprocessing
    └── partitions/          # Saved per-client data splits for Scenario 1
```

### 3.1 Scope: Starting Infrastructure vs. CNN Body

This plan describes the **full** SSFL system, but the **first implementation pass intentionally ships only the starting infrastructure**. The scope of that first pass is:

**Fully implemented in the first pass:**
- `data_preparation.py` — loading, normalization, splitting, sharding, partition I/O.
- `client.py` — `SSFLClient` class with the complete 5-step orchestration in `fit()`, `get_parameters`/`set_parameters` repurposed for hard labels, and `evaluate()` shell.
- `strategy.py` — `SSFLStrategy` including `configure_fit`, `aggregate_fit`, `vote_mechanism`, `configure_evaluate`, `aggregate_evaluate`.
- `server.py` — `start_server` and the `build_eval_fn` closure (with the eval model built via the stub factories, so the closure can be constructed even though it will raise if actually invoked).
- `main.py` — `parse_arguments`, `setup_environment`, `run_server`, `run_client`, `main`.
- `utils.py`, `config.py` — all helpers and constants.

**Deliberately stubbed in the first pass — the "undeclared entry point":**
- `model.py` — `TrafficCNN`, `build_classifier`, and `build_discriminator` are **declared with their final signatures** but their bodies raise `NotImplementedError("CNN body deferred to implementation pass 2")`. The surrounding code calls these factories exactly as it will in the finished system, so the wiring is exercised, but no CNN layers are laid down yet.
- The per-step training methods on the client (`train_classifier`, `train_discriminator`, `run_distillation`, `compute_confidence_scores`, `filter_and_predict`) may also be stubbed to raise `NotImplementedError`, since they cannot run meaningfully without a real model.

**Rule of thumb:** if a function's body depends on an actual CNN forward pass or backward pass, it can stub; if it is orchestration, I/O, voting, partitioning, or plumbing, it ships live. The point of the first pass is to prove the 27-client Flower loop, the label-over-parameters contract, and the voting logic all work end-to-end with a trivially placeholder model — so that when the real `TrafficCNN` is dropped in at `model.py`, nothing else has to move.

---

## 4. Module: `data_preparation.py`

**Purpose:** This module handles every data-related operation — loading raw N-BaIoT files, normalizing features, reshaping samples into 2D matrices, creating the mini dataset, splitting into private/open/test sets, and applying the Scenario 1 shard partitioning. It is run **once** before training and saves partitions to disk so that each client process can load its own slice.

---

### 4.1 `load_device_csvs(device_dir: str) -> pd.DataFrame`

**Purpose:** Loads all CSV files from a single IoT device directory and combines them into one DataFrame, attaching a numeric class label to each row based on which CSV file it came from (each file corresponds to one traffic category).

**Parameters:**
- `device_dir: str` — Absolute path to the folder containing CSV files for one IoT device (e.g., `./data/raw/Danmini_Doorbell/`). Each CSV file inside this folder represents one traffic category (one benign file + multiple attack files).

**Returns:** `pd.DataFrame` — A single DataFrame with all samples from all traffic categories for this device. Contains 115 feature columns plus a `label` column. Labels use the **global label space 0–10** (mapped from filename → global class index via a fixed lookup, so that, e.g., "mirai_udp" always maps to the same integer across every device that has it). Devices missing a class simply never produce rows with that label.

**Internal Variables:**
- `all_frames: List[pd.DataFrame]` — A list that accumulates one DataFrame per CSV file before concatenation. Initially empty.
- `csv_files: List[str]` — Sorted list of all `.csv` file paths found inside `device_dir`. Sorted alphabetically so that iteration order is deterministic and reproducible.
- `class_name: str` — The traffic category name derived from the CSV filename (e.g., `"benign"`, `"gafgyt_combo"`, `"mirai_udp"`).
- `global_label: int` — The integer class label in the **global 0–10 label space**, looked up from `class_name` via the shared `CLASS_NAME_TO_GLOBAL_ID` mapping in `config.py`. This guarantees that the same attack family receives the same integer across all 9 devices, even when individual devices are missing some classes.
- `df: pd.DataFrame` — Temporary variable holding the DataFrame loaded from a single CSV file during each loop iteration.
- `device_df: pd.DataFrame` — The final concatenated DataFrame after all CSV files for this device are loaded and labeled.

**How it works:** It scans `device_dir` for `.csv` files, loads each one with `pd.read_csv()`, extracts the class name from the filename, looks up `global_label` via the shared name→id mapping, assigns `global_label` as the `label` column for every row in that file, appends to `all_frames`, then concatenates everything into `device_df` and resets the index.

---

### 4.2 `build_mini_nbaiot(raw_data_dir: str, samples_per_class: int = 1000) -> pd.DataFrame`

**Purpose:** Constructs the mini-N-BaIoT dataset by loading all 9 device folders, then sampling exactly `samples_per_class` records from each `(device, class)` combination. This reduces the dataset to a manageable size while preserving class diversity across devices.

**Parameters:**
- `raw_data_dir: str` — Path to the root directory containing one subfolder per IoT device (9 subfolders total).
- `samples_per_class: int` — How many records to keep per traffic class per device. Default is `1000` as specified in the paper.

**Returns:** `pd.DataFrame` — The complete mini-N-BaIoT dataset with all devices and all classes sampled down to `samples_per_class` each. Also contains a `device_id` column (integer 0–8) identifying which device each sample comes from.

**Internal Variables:**
- `device_dirs: List[str]` — List of all subdirectory paths found inside `raw_data_dir`, one per device. Sorted for reproducibility.
- `device_id: int` — Loop counter tracking which device is currently being processed (0 through 8).
- `device_df: pd.DataFrame` — Raw DataFrame for the current device, returned by `load_device_csvs()`.
- `sampled_frames: List[pd.DataFrame]` — Accumulates the sampled subsets from all devices before final concatenation.
- `class_labels: np.ndarray` — Array of unique integer labels found in `device_df['label']`, used to iterate over each class.
- `class_subset: pd.DataFrame` — Temporary variable holding all rows belonging to one specific class within the current device.
- `sampled_class: pd.DataFrame` — The result of calling `.sample(n=samples_per_class)` on `class_subset`. If a class has fewer than `samples_per_class` rows, sampling is done with replacement (`replace=True`).
- `mini_df: pd.DataFrame` — Final concatenated mini dataset with index reset.

---

### 4.3 `normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, dict]`

**Purpose:** Applies min-max normalization to all 115 feature columns, scaling each feature independently to the range [0, 1]. The normalization parameters (min and max per feature) are computed on the training data and returned so the same parameters can be applied to the test set.

**Parameters:**
- `df: pd.DataFrame` — The DataFrame containing raw feature values. Must include all columns listed in `feature_cols`.
- `feature_cols: List[str]` — List of exactly 115 column names corresponding to the traffic features. Label and device_id columns are excluded and left unchanged.

**Returns:** `Tuple[pd.DataFrame, dict]`
- The first element is the normalized DataFrame with feature values in [0, 1].
- The second element is a `dict` called `norm_params` with keys `'min'` and `'max'`, each mapping to a `pd.Series` of per-column min/max values. This dict is saved to disk so test data can be normalized with the same parameters.

**Internal Variables:**
- `df_copy: pd.DataFrame` — A deep copy of `df` created at the start to avoid mutating the input.
- `feature_min: pd.Series` — Per-column minimum values computed from `df[feature_cols].min()`. Shape: `(115,)`.
- `feature_max: pd.Series` — Per-column maximum values computed from `df[feature_cols].max()`. Shape: `(115,)`.
- `range_vals: pd.Series` — `feature_max - feature_min` for each column. Any zero-range column (constant feature) gets replaced with 1.0 to prevent division by zero.
- `norm_params: dict` — Dictionary storing `feature_min` and `feature_max` for later use on test/open data.

**Formula applied:** For each feature column `f` and each value `x`: `x_normalized = (x - min_f) / (max_f - min_f)`

---

### 4.4 `reshape_sample_to_2d(sample_vector: np.ndarray) -> np.ndarray`

**Purpose:** Converts one flat 115-feature sample vector into a 2D matrix of shape `(23, 5)` so it can be processed by the 1D convolutional layers of the CNN. The 115 features naturally divide into 5 groups of 23, each corresponding to one time window (100ms, 500ms, 1.5s, 10s, 1min).

**Parameters:**
- `sample_vector: np.ndarray` — A 1D numpy array of shape `(115,)` containing the normalized feature values for one traffic sample.

**Returns:** `np.ndarray` — A 2D numpy array of shape `(23, 5)`. Column `j` contains features from time window `j` (i.e., features at indices `j, j+23, j+46, j+69, j+92` for `j` in `0..22`).

**Internal Variables:**
- `matrix: np.ndarray` — A pre-allocated array of shape `(23, 5)` filled with zeros, into which feature values are placed.
- `time_window_idx: int` — Loop variable ranging from 0 to 4, indexing each of the 5 time windows.
- `feature_offset: int` — Computed as `time_window_idx * 23`, the starting index in `sample_vector` for the current time window block.

**Reshaping logic:** Feature `x_{i, j*23 + k}` from the flat vector (where `j` is time window index and `k` is the within-window index) maps to `matrix[k, j]`.

---

### 4.5 `apply_2d_reshape_to_dataset(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]`

**Purpose:** Applies `reshape_sample_to_2d()` to every sample in the dataset and returns the full feature tensor and label array ready for use with PyTorch DataLoaders.

**Parameters:**
- `df: pd.DataFrame` — Normalized DataFrame containing all samples.
- `feature_cols: List[str]` — The 115 feature column names.

**Returns:** `Tuple[np.ndarray, np.ndarray]`
- `X: np.ndarray` — Feature tensor of shape `(N, 23, 5)` where `N` is the number of samples. This shape matches what the CNN input layer expects.
- `y: np.ndarray` — Label array of shape `(N,)` containing integer class labels. For the open dataset, this is an array of `-1` values (labels unknown).

**Internal Variables:**
- `feature_matrix: np.ndarray` — Raw values of the feature columns extracted from `df` as a numpy array of shape `(N, 115)`.
- `reshaped_list: List[np.ndarray]` — List collecting the `(23, 5)` matrix for each sample before stacking.
- `X: np.ndarray` — Final stacked array of shape `(N, 23, 5)`.
- `y: np.ndarray` — Integer label array extracted from `df['label']` column. Converted to `np.int64`.

---

### 4.6 `split_private_open_test(mini_df: pd.DataFrame, private_ratio: float = 0.70, open_ratio: float = 0.10, test_ratio: float = 0.20, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

**Purpose:** Divides the full mini-N-BaIoT dataset into three completely disjoint splits: private (for client training), open (for distillation, labels stripped), and test (for server evaluation). The split is **stratified** by device and class to ensure all three sets contain samples from every category.

**Parameters:**
- `mini_df: pd.DataFrame` — The complete mini-N-BaIoT dataset.
- `private_ratio: float` — Fraction of data to assign to the private training pool. Default `0.70`.
- `open_ratio: float` — Fraction of data to assign to the open (unlabeled distillation) pool. Default `0.10`.
- `test_ratio: float` — Fraction of data to assign to the test set. Default `0.20`. Must sum to `1.0` with the other two.
- `random_seed: int` — Random seed for reproducible shuffling. Default `42`.

**Returns:** `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- `private_df: pd.DataFrame` — Private labeled data (70%). Contains `label` and `device_id` columns.
- `open_df: pd.DataFrame` — Open unlabeled data (10%). The `label` column is **dropped** before returning.
- `test_df: pd.DataFrame` — Test data (20%). Retains `label` column for server evaluation.

**Internal Variables:**
- `rng: np.random.Generator` — Random number generator initialized with `random_seed` for reproducibility.
- `private_parts: List[pd.DataFrame]` — Accumulates private split portions per `(device_id, class)` group.
- `open_parts: List[pd.DataFrame]` — Accumulates open split portions per `(device_id, class)` group.
- `test_parts: List[pd.DataFrame]` — Accumulates test split portions per `(device_id, class)` group.
- `group: pd.DataFrame` — Temporary variable holding all samples for one `(device_id, label)` group during stratified splitting.
- `n: int` — Total number of samples in the current group.
- `n_private: int` — Number of samples to assign to private split from this group. Computed as `int(n * private_ratio)`.
- `n_open: int` — Number of samples to assign to open split from this group. Computed as `int(n * open_ratio)`.
- `n_test: int` — Remaining samples assigned to test split: `n - n_private - n_open`.
- `shuffled_group: pd.DataFrame` — The group with rows shuffled using `rng`.

---

### 4.7 `partition_scenario1(private_df: pd.DataFrame, target_device_id: int, k_di: int = 3) -> List[pd.DataFrame]`

**Purpose:** Implements the Scenario 1 shard-based non-IID partitioning from the paper. For one specific device's private data, it creates `k_di` client datasets by sorting samples by label, dividing into `2 * k_di` equal shards, and assigning exactly 2 consecutive shards to each client.

**Parameters:**
- `private_df: pd.DataFrame` — The full private dataset (all clients, all devices). This function filters to `target_device_id` internally.
- `target_device_id: int` — Integer identifier (0–8) of the IoT device whose data is being partitioned. Named distinctly from the DataFrame column `device_id` to avoid self-referential filter bugs.
- `k_di: int` — Number of clients for this device. In Scenario 1, always `3`.

**Returns:** `List[pd.DataFrame]` — A list of `k_di` DataFrames, where index `i` is the private dataset for client `i` of this device. Each DataFrame contains its `label` column but represents only 2 out of `2*k_di` label shards.

**Internal Variables:**
- `device_data: pd.DataFrame` — Subset of `private_df` filtered to rows where `private_df['device_id'] == target_device_id`. This is the pool of labeled samples for one specific device.
- `sorted_data: pd.DataFrame` — `device_data` sorted by the `label` column in ascending order. This is the foundation of the shard construction — consecutive rows will belong to the same or adjacent classes.
- `n_total: int` — Total number of samples in `device_data` (= `len(sorted_data)`).
- `shard_size: int` — Size of each individual shard, computed as `n_total // (2 * k_di)`. Any remainder samples are discarded to keep shards equal-sized.
- `n_shards: int` — Total number of shards, equal to `2 * k_di` (= 6 in Scenario 1).
- `shards: List[pd.DataFrame]` — A list of `n_shards` DataFrames, each of length `shard_size`, created by slicing `sorted_data` at regular intervals.
- `client_datasets: List[pd.DataFrame]` — The output list. Built by iterating over clients (0, 1, 2) and concatenating shards `[2*i, 2*i+1]` for client `i`.
- `client_df: pd.DataFrame` — Temporary variable holding the concatenation of the two shards for one client before appending to `client_datasets`.

---

### 4.8 `build_all_client_partitions(private_df: pd.DataFrame, k_di: int = 3) -> Dict[int, pd.DataFrame]`

**Purpose:** Calls `partition_scenario1()` for all 9 devices and assembles a global mapping from **global client ID** (0–26) to that client's private DataFrame. The global client ID is computed as `device_id * k_di + local_client_id`.

**Parameters:**
- `private_df: pd.DataFrame` — The full private dataset for all devices combined.
- `k_di: int` — Number of clients per device. Always `3` for Scenario 1.

**Returns:** `Dict[int, pd.DataFrame]` — Dictionary mapping global client ID (integer 0–26) to the client's private labeled DataFrame. This dictionary is serialized to disk so each client process can look up its own partition.

**Internal Variables:**
- `all_partitions: Dict[int, pd.DataFrame]` — The output dictionary, built up during the loop.
- `device_id: int` — Loop variable from 0 to 8 (one iteration per IoT device); passed as `target_device_id` into `partition_scenario1`.
- `device_partitions: List[pd.DataFrame]` — Result of `partition_scenario1(private_df, target_device_id=device_id, k_di=k_di)` for the current device — a list of 3 DataFrames.
- `local_client_id: int` — Loop variable from 0 to `k_di-1` (one iteration per client within a device).
- `global_client_id: int` — Computed as `device_id * k_di + local_client_id`. This is the unique identifier used across the entire federated system.

---

### 4.9 `save_partitions(all_partitions: Dict[int, pd.DataFrame], open_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None`

**Purpose:** Serializes all client data partitions, the open dataset, and the test dataset to disk as pickle files. Each client's data is saved separately so that a Flower client process can load only its own partition without loading the full dataset.

**Parameters:**
- `all_partitions: Dict[int, pd.DataFrame]` — The full client partition mapping from `build_all_client_partitions()`.
- `open_df: pd.DataFrame` — The unlabeled open dataset (labels already stripped).
- `test_df: pd.DataFrame` — The test dataset with labels.
- `output_dir: str` — Directory where files will be saved. Created if it doesn't exist.

**Returns:** `None`. Side effect: writes files to disk.

**Internal Variables:**
- `client_id: int` — Loop variable iterating over keys of `all_partitions`.
- `file_path: str` — Constructed path like `{output_dir}/client_{client_id}_private.pkl` for each client.
- `open_path: str` — Path `{output_dir}/open_data.pkl` for the open dataset.
- `test_path: str` — Path `{output_dir}/test_data.pkl` for the test dataset.

---

### 4.10 `load_client_partition(client_id: int, partition_dir: str) -> Tuple[np.ndarray, np.ndarray]`

**Purpose:** Loads the private partition for a specific client from disk and returns it as ready-to-use numpy arrays (already reshaped to 2D). Called once when a Flower client process starts up.

**Parameters:**
- `client_id: int` — The global client ID (0–26) of the client loading its data.
- `partition_dir: str` — Directory where partition files are stored.

**Returns:** `Tuple[np.ndarray, np.ndarray]`
- `X_private: np.ndarray` — Feature tensor of shape `(N_client, 23, 5)`.
- `y_private: np.ndarray` — Label array of shape `(N_client,)` with integer class labels.

**Internal Variables:**
- `file_path: str` — Constructed path to this client's partition pickle file.
- `client_df: pd.DataFrame` — The loaded DataFrame from the pickle file.
- `feature_cols: List[str]` — The 115 feature column names loaded from a shared config.

---

### 4.11 `load_open_data(partition_dir: str) -> np.ndarray`

**Purpose:** Loads the shared unlabeled open dataset from disk and returns it as a numpy feature tensor. This is called by every client process at startup — all clients share the same open data.

**Parameters:**
- `partition_dir: str` — Directory where `open_data.pkl` is stored.

**Returns:** `np.ndarray` — Feature tensor `X_open` of shape `(N_open, 23, 5)` where `N_open` is the total number of open samples (approximately 10% of mini-N-BaIoT).

**Internal Variables:**
- `open_df: pd.DataFrame` — DataFrame loaded from `open_data.pkl` (no label column).
- `feature_cols: List[str]` — The 115 feature column names.
- `X_open: np.ndarray` — Reshaped feature array returned to the caller.

---

### 4.12 `create_torch_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader`

**Purpose:** Wraps numpy arrays into a PyTorch `TensorDataset` and creates a `DataLoader` for batched training. This is a utility called whenever a model needs to be trained on a dataset.

**Parameters:**
- `X: np.ndarray` — Feature array, typically shape `(N, 23, 5)`.
- `y: np.ndarray` — Label array of shape `(N,)` with integer labels.
- `batch_size: int` — Number of samples per mini-batch. Set to `100` as per the paper.
- `shuffle: bool` — Whether to shuffle the data before each epoch. Default `True` for training, `False` for inference.

**Returns:** `torch.utils.data.DataLoader` — A DataLoader that yields `(feature_batch, label_batch)` tuples of tensors.

**Internal Variables:**
- `X_tensor: torch.Tensor` — Converted from `X` using `torch.FloatTensor()`. Shape `(N, 23, 5)`.
- `y_tensor: torch.Tensor` — Converted from `y` using `torch.LongTensor()`. Shape `(N,)`.
- `dataset: TensorDataset` — Wraps `X_tensor` and `y_tensor` together.
- `loader: DataLoader` — The returned DataLoader object.

---

## 5. Module: `model.py`

**Purpose:** Defines the CNN architecture used for both the classifier and the discriminator. The paper states both networks share the same architecture except for the output layer — the classifier outputs `L` class probabilities (L=11) while the discriminator outputs 2 probabilities (familiar vs unfamiliar).

---

### 5.1 Class: `TrafficCNN(nn.Module)`

**Purpose:** The core CNN model. Used as both the classifier (output size = number of traffic classes) and the discriminator (output size = 2). The architecture has 8 Conv1D layers followed by 2 fully-connected layers.

**Constructor: `__init__(self, num_classes: int)`**

**Parameters:**
- `num_classes: int` — Number of output neurons. Pass `11` for classifier (11 traffic classes in N-BaIoT 11-class devices), pass `2` for discriminator (familiar/unfamiliar).

**Internal Attributes (set in `__init__`):**
- `self.num_classes: int` — Stored copy of `num_classes` for reference.

- `self.conv_block1: nn.Sequential` — First convolutional block containing:
  - `nn.Conv1d(in_channels=23, out_channels=64, kernel_size=3, padding=1)` — Takes the 23-row input (23 features per time window), outputs 64 feature maps. `padding=1` keeps the temporal dimension equal (size 5 → size 5).
  - `nn.BatchNorm1d(64)` — Normalizes activations across the batch for training stability.
  - `nn.ReLU()` — Non-linear activation.

- `self.conv_block2: nn.Sequential` — Second convolutional block, same structure as block 1: Conv1d(64→64, kernel=3), BatchNorm1d(64), ReLU.

- `self.conv_block3: nn.Sequential` — Third convolutional block: Conv1d(64→64, kernel=3), BatchNorm1d(64), ReLU.

- `self.conv_block4: nn.Sequential` — Fourth convolutional block: Conv1d(64→64, kernel=3), BatchNorm1d(64), ReLU. After this block, the feature maps have shape `(batch, 64, 5)`.

- `self.conv_block5: nn.Sequential` — Fifth convolutional block, upgrades to 128 channels: Conv1d(64→128, kernel=3, padding=1), BatchNorm1d(128), ReLU.

- `self.conv_block6: nn.Sequential` — Sixth convolutional block: Conv1d(128→128, kernel=3, padding=1), BatchNorm1d(128), ReLU.

- `self.conv_block7: nn.Sequential` — Seventh convolutional block: Conv1d(128→128, kernel=3, padding=1), BatchNorm1d(128), ReLU.

- `self.conv_block8: nn.Sequential` — Eighth convolutional block: Conv1d(128→128, kernel=3, padding=1), BatchNorm1d(128), ReLU. After this, feature maps have shape `(batch, 128, 5)`.

- `self.flatten: nn.Flatten` — Flattens `(batch, 128, 5)` into `(batch, 640)`.

- `self.fc1: nn.Linear(640, 128)` — First fully-connected layer reducing from 640 to 128 neurons.

- `self.fc1_relu: nn.ReLU()` — Activation after `fc1`.

- `self.dropout: nn.Dropout(p=0.5)` — Dropout for regularization. Applied after `fc1_relu`.

- `self.fc2: nn.Linear(128, num_classes)` — Output layer. For the classifier: `num_classes=11`. For the discriminator: `num_classes=2`.

**Note on Conv1d input format:** The input tensor to the model must have shape `(batch_size, 23, 5)`. The `23` dimension is treated as `in_channels` and the `5` time windows are treated as the 1D sequence length. This is consistent with Table I in the paper.

---

**Method: `forward(self, x: torch.Tensor) -> torch.Tensor`**

**Purpose:** Defines the forward pass of the network. Called automatically by PyTorch during training and inference.

**Parameters:**
- `x: torch.Tensor` — Input batch tensor of shape `(batch_size, 23, 5)`. The 23 channels correspond to the 23 within-window features, and 5 is the number of time windows.

**Returns:** `torch.Tensor` — Raw logits (pre-softmax scores) of shape `(batch_size, num_classes)`. During training, cross-entropy loss is applied to these raw logits. During inference, softmax is applied to convert to probabilities.

**Internal Variables (local to forward):**
- `out: torch.Tensor` — Running variable tracking the tensor as it flows through each layer. Initialized to `x`, then progressively transformed by each convolutional block and FC layer.

**Flow:**
```
x → conv_block1 → conv_block2 → conv_block3 → conv_block4
  → conv_block5 → conv_block6 → conv_block7 → conv_block8
  → flatten → fc1 → fc1_relu → dropout → fc2 → return
```

---

### 5.2 `build_classifier(num_classes: int, device: torch.device) -> TrafficCNN`

**Purpose:** Factory function that instantiates a `TrafficCNN` configured as a **classifier** and moves it to the specified compute device (CPU or GPU).

**Parameters:**
- `num_classes: int` — Number of traffic classes for this model instance (11 for 11-class devices).
- `device: torch.device` — The PyTorch device to place the model on (e.g., `torch.device('cuda')` or `torch.device('cpu')`).

**Returns:** `TrafficCNN` — A newly initialized classifier model on the specified device.

**Internal Variables:**
- `model: TrafficCNN` — Instantiated with `TrafficCNN(num_classes=num_classes)`.

---

### 5.3 `build_discriminator(device: torch.device) -> TrafficCNN`

**Purpose:** Factory function that instantiates a `TrafficCNN` configured as a **discriminator** (2 output classes: familiar=index 0, unfamiliar=index 1).

**Parameters:**
- `device: torch.device` — PyTorch device to place the model on.

**Returns:** `TrafficCNN` — A newly initialized discriminator model with `num_classes=2`.

**Internal Variables:**
- `model: TrafficCNN` — Instantiated with `TrafficCNN(num_classes=2)`.

---

### 5.4 `get_model_parameters(model: TrafficCNN) -> List[np.ndarray]`

**Purpose:** Extracts the full model state — **both learnable parameters and non-learnable buffers** (e.g., BatchNorm `running_mean` / `running_var`) — as a list of numpy arrays. This is used for **local checkpointing, deterministic seed-loading, and server-side eval model snapshots**. It is **not** used for the Flower channel in our SSFL system: the Flower `get_parameters` in `client.py` sends hard labels, not model weights.

**Parameters:**
- `model: TrafficCNN` — The model whose state to extract.

**Returns:** `List[np.ndarray]` — One array per entry in `model.state_dict()`, in insertion order, each converted from `torch.Tensor` to `np.ndarray`.

**Internal Variables:**
- `state: Dict[str, torch.Tensor]` — Result of `model.state_dict()`. Includes BatchNorm running statistics, not just learnable parameters.
- `params: List[np.ndarray]` — Built by iterating `state.values()` and calling `.cpu().detach().numpy()` on each tensor.

**Why `state_dict()` and not `model.parameters()`:** `model.parameters()` only yields tensors with `requires_grad=True`. In `TrafficCNN`, the BatchNorm layers own running statistics that are **buffers**, not parameters. Snapshotting via `parameters()` would silently drop those buffers, and restoring would leave BatchNorm in a mismatched state.

---

### 5.5 `set_model_parameters(model: TrafficCNN, parameters: List[np.ndarray]) -> None`

**Purpose:** Loads a list of numpy arrays back into the model's full state (learnable parameters **and** buffers). Used to restore a model snapshot previously saved by `get_model_parameters`.

**Parameters:**
- `model: TrafficCNN` — The model whose state to update in-place.
- `parameters: List[np.ndarray]` — List of numpy arrays matching `model.state_dict()` key order exactly.

**Returns:** `None`. Mutates `model` in place.

**Internal Variables:**
- `state_keys: List[str]` — The keys of `model.state_dict()` in insertion order.
- `new_state: Dict[str, torch.Tensor]` — Built by zipping `state_keys` with `parameters`, wrapping each numpy array in `torch.tensor(...)` with the correct dtype.
- The final load call is `model.load_state_dict(new_state, strict=True)`, which fails loudly on any mismatch rather than silently half-updating the model.

---

## 6. Module: `client.py`

**Purpose:** Defines the `SSFLClient` class which inherits from `flwr.client.NumPyClient`. Each instance of this class represents one IoT device client. It holds the client's private dataset, the shared open dataset, the classifier model, and the discriminator model. The `fit()` method orchestrates all 5 SSFL steps for one communication round.

---

### 6.1 Class: `SSFLClient(flwr.client.NumPyClient)`

**Constructor: `__init__(self, client_id, X_private, y_private, X_open, num_classes, device, learning_rate, batch_size, classifier_epochs, discriminator_epochs)`**

**Purpose:** Initializes the client with its private data, open data, two CNN models, and training configuration. Called once when the Flower client process starts.

**Parameters:**
- `client_id: int` — The global client ID (0–26). Used for logging and identification.
- `X_private: np.ndarray` — Private labeled feature array, shape `(N_private, 23, 5)`. This data **never leaves the client**.
- `y_private: np.ndarray` — Private label array, shape `(N_private,)`. Contains integer class labels 0–10.
- `X_open: np.ndarray` — Shared unlabeled open data feature array, shape `(N_open, 23, 5)`. Same array is held by all clients.
- `num_classes: int` — Number of traffic classes for this client's device (11 for most devices).
- `device: torch.device` — PyTorch compute device (CPU or CUDA).
- `learning_rate: float` — Adam optimizer learning rate. Paper uses `0.0001`.
- `batch_size: int` — Mini-batch size for all training loops. Paper uses `100`.
- `classifier_epochs: int` — Number of local training epochs for the classifier per round. Paper uses `5`.
- `discriminator_epochs: int` — Number of training epochs for the discriminator per round.

**Internal Attributes (set in `__init__`):**
- `self.client_id: int` — Stored client identifier.
- `self.X_private: np.ndarray` — Stored private feature data.
- `self.y_private: np.ndarray` — Stored private label data.
- `self.X_open: np.ndarray` — Stored open dataset features.
- `self.N_open: int` — Total number of open samples. `len(X_open)`. Used to size the label upload array.
- `self.num_classes: int` — Stored class count.
- `self.device: torch.device` — Stored compute device.
- `self.learning_rate: float` — Stored learning rate.
- `self.batch_size: int` — Stored batch size.
- `self.classifier_epochs: int` — Stored number of classifier training epochs per round.
- `self.discriminator_epochs: int` — Stored discriminator training epochs per round.
- `self.classifier: TrafficCNN` — The classifier model. Created via `build_classifier(num_classes, device)`. Starts randomly initialized.
- `self.discriminator: TrafficCNN` — The discriminator model. Created via `build_discriminator(device)`. Starts randomly initialized. **Re-initialized fresh each round** because the definition of "familiar" vs "unfamiliar" data changes as the classifier improves.
- `self.classifier_optimizer: torch.optim.Adam` — Adam optimizer for the classifier. Initialized with `lr=learning_rate`. Persists across rounds to benefit from optimizer momentum.
- `self.current_round: int` — Tracks the current communication round number. Incremented in `fit()`. Used for logging.
- `self.global_labels: np.ndarray` — The most recently received global labels from the server. Shape `(N_open,)` with integers 0–10. Initially `None`. Updated at the start of each `fit()` call when the server sends new global labels.

---

### 6.2 `train_classifier(self) -> float`

**Purpose:** Executes Step 1 of the SSFL algorithm. Trains the classifier model on the client's private labeled data for `self.classifier_epochs` epochs using standard supervised cross-entropy loss. This makes the classifier learn the local traffic patterns.

**Parameters:** None (uses `self` attributes).

**Returns:** `float` — The average cross-entropy training loss over the last epoch. Used for logging only.

**Internal Variables:**
- `private_loader: DataLoader` — DataLoader wrapping `self.X_private` and `self.y_private`, with `batch_size=self.batch_size` and `shuffle=True`. Created fresh each call.
- `criterion: nn.CrossEntropyLoss` — The loss function. `CrossEntropyLoss` combines log-softmax and NLL loss, so the model's raw logits are passed directly.
- `epoch: int` — Loop variable from 0 to `self.classifier_epochs - 1`.
- `total_loss: float` — Accumulates loss across all batches in one epoch. Reset each epoch.
- `n_batches: int` — Counts the number of batches processed. Used to compute average loss.
- `X_batch: torch.Tensor` — Feature batch of shape `(batch_size, 23, 5)` moved to `self.device`.
- `y_batch: torch.Tensor` — Label batch of shape `(batch_size,)` moved to `self.device`.
- `logits: torch.Tensor` — Output of `self.classifier(X_batch)`, shape `(batch_size, num_classes)`. Raw, unnormalized scores.
- `loss: torch.Tensor` — Scalar tensor from `criterion(logits, y_batch)`.
- `avg_loss: float` — `total_loss / n_batches` for the final epoch. This is the return value.

**Side effects:** Updates `self.classifier` weights via backpropagation and `self.classifier_optimizer`.

---

### 6.3 `compute_confidence_scores(self) -> np.ndarray`

**Purpose:** Runs the trained classifier in inference mode over all open samples and computes the **confidence score** for each sample — defined as the maximum predicted probability across all classes (i.e., `max(softmax(logits))`). This score reflects how confidently the classifier can classify each open sample.

**Parameters:** None.

**Returns:** `np.ndarray` — Confidence score array of shape `(N_open,)` containing float values in [0, 1]. A high score means the classifier is very confident; a low score suggests the sample type is unfamiliar to this client.

**Internal Variables:**
- `open_loader: DataLoader` — DataLoader wrapping `self.X_open` with dummy labels (-1), `batch_size=self.batch_size`, `shuffle=False`. Order must be preserved because confidence scores are indexed by position.
- `confidence_scores: List[float]` — Accumulates confidence values as batches are processed.
- `X_batch: torch.Tensor` — Open data feature batch moved to `self.device`.
- `logits: torch.Tensor` — Raw classifier output for the batch, shape `(batch_size, num_classes)`.
- `probs: torch.Tensor` — Softmax probabilities, shape `(batch_size, num_classes)`. Computed with `torch.softmax(logits, dim=1)`.
- `max_probs: torch.Tensor` — Maximum probability per sample, shape `(batch_size,)`. Computed with `probs.max(dim=1).values`.
- `scores_np: np.ndarray` — Numpy version of `max_probs.cpu().detach()`.

**Implementation note:** Wrapped in `with torch.no_grad():` to disable gradient tracking for efficiency.

---

### 6.4 `compute_confidence_threshold(self, confidence_scores: np.ndarray) -> float`

**Purpose:** Computes the adaptive confidence threshold `θ` for this client. The paper specifies that each client uses the **median** of its own set of confidence scores rather than a fixed value. This adaptive threshold performs better than fixed values (0.7, 0.8, 0.9) especially for clients with fewer labeled classes.

**Parameters:**
- `confidence_scores: np.ndarray` — Shape `(N_open,)` array of confidence scores from `compute_confidence_scores()`.

**Returns:** `float` — The median confidence score, which serves as the threshold `θ`. Samples with a confidence score below this threshold are labeled as "unfamiliar."

**Internal Variables:**
- `threshold: float` — Computed via `float(np.median(confidence_scores))`.

---

### 6.5 `build_discriminator_dataset(self, confidence_scores: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]`

**Purpose:** Constructs the training dataset `D^{k,d}` for the discriminator. This dataset contains two types of samples: (1) open samples whose confidence score is below `threshold` → labeled as "unfamiliar" (label 1), and (2) all private samples → labeled as "familiar" (label 0). The discriminator learns to separate known traffic patterns from unknown ones.

**Parameters:**
- `confidence_scores: np.ndarray` — Shape `(N_open,)` confidence score array for all open samples.
- `threshold: float` — The adaptive threshold `θ` below which an open sample is considered unfamiliar.

**Returns:** `Tuple[np.ndarray, np.ndarray]`
- `X_disc: np.ndarray` — Feature array for discriminator training, shape `(N_unfamiliar + N_private, 23, 5)`.
- `y_disc: np.ndarray` — Binary label array for discriminator training, shape `(N_unfamiliar + N_private,)`. Values are 0 (familiar) or 1 (unfamiliar).

**Internal Variables:**
- `unfamiliar_mask: np.ndarray` — Boolean mask of shape `(N_open,)`. `True` at index `j` if `confidence_scores[j] < threshold`. Identifies which open samples the classifier is uncertain about.
- `X_unfamiliar: np.ndarray` — Open samples where `unfamiliar_mask` is `True`. These form the "unfamiliar" portion of the discriminator dataset.
- `y_unfamiliar: np.ndarray` — All-ones array of shape `(len(X_unfamiliar),)` — the "unfamiliar" discriminator label.
- `X_private_familiar: np.ndarray` — All private samples `self.X_private`. The client treats every sample it was trained on as "familiar."
- `y_private_familiar: np.ndarray` — All-zeros array of shape `(N_private,)` — the "familiar" discriminator label.
- `X_disc: np.ndarray` — Concatenation of `X_unfamiliar` and `X_private_familiar`.
- `y_disc: np.ndarray` — Concatenation of `y_unfamiliar` and `y_private_familiar`.

**Design note:** High-confidence open samples (where `unfamiliar_mask` is `False`) are deliberately **not** added to the "familiar" side of the discriminator dataset. The paper treats only private data as authoritatively familiar and only low-confidence open data as unfamiliar; high-confidence open samples are excluded from discriminator training to avoid feeding the classifier's own (possibly wrong) confident predictions back into its own decision boundary.

**Note:** `y_disc` uses integers 0 and 1, not the one-hot vectors shown in the paper's equations (the paper shows one-hot `[1,0]^T` for familiar and `[0,1]^T` for unfamiliar). Our implementation uses integer labels because PyTorch's `CrossEntropyLoss` handles this automatically.

---

### 6.6 `train_discriminator(self, X_disc: np.ndarray, y_disc: np.ndarray) -> float`

**Purpose:** Trains the discriminator model on the dataset built by `build_discriminator_dataset()`. The discriminator learns to output "familiar" (class 0) for traffic the client knows and "unfamiliar" (class 1) for traffic it has never seen.

**Parameters:**
- `X_disc: np.ndarray` — Feature array for discriminator training from `build_discriminator_dataset()`.
- `y_disc: np.ndarray` — Binary label array (0=familiar, 1=unfamiliar).

**Returns:** `float` — Average training loss over the last epoch.

**Internal Variables:**
- `disc_loader: DataLoader` — DataLoader for `(X_disc, y_disc)`, with `batch_size=self.batch_size` and `shuffle=True`.
- `disc_optimizer: torch.optim.Adam` — A **fresh** Adam optimizer created for the discriminator each round (unlike the classifier optimizer, the discriminator is re-trained from scratch each round).
- `criterion: nn.CrossEntropyLoss` — Binary cross-entropy equivalent for 2-class classification.
- `epoch: int` — Loop variable from 0 to `self.discriminator_epochs - 1`.
- `total_loss: float` — Accumulated loss per epoch.
- `n_batches: int` — Batch counter for averaging.
- `X_batch: torch.Tensor` — Feature batch moved to `self.device`.
- `y_batch: torch.Tensor` — Binary label batch moved to `self.device`.
- `logits: torch.Tensor` — Discriminator output, shape `(batch_size, 2)`.
- `loss: torch.Tensor` — Cross-entropy loss scalar.

**Side effects:** Updates `self.discriminator` weights.

---

### 6.7 `filter_and_predict(self) -> np.ndarray`

**Purpose:** Implements Step 3 of the SSFL algorithm — "Filter and Upload." For each open sample, the trained discriminator decides whether it is "familiar" or "unfamiliar." Familiar samples get a hard label from the classifier (the `argmax` of classifier output), while unfamiliar samples are assigned `-1`. The resulting array is what gets uploaded to the server.

**Parameters:** None.

**Returns:** `np.ndarray` — Hard label array of shape `(N_open,)` with integer values. Each value is either:
- An integer in `[0, num_classes-1]` — the classifier's predicted class for a familiar sample.
- `-1` — the sample is unfamiliar to this client and no prediction is contributed.

**Internal Variables:**
- `hard_labels: np.ndarray` — Output array initialized to `-1` for all positions (assuming all unfamiliar).
- `open_loader: DataLoader` — DataLoader for `self.X_open`, no shuffle, preserves order.
- `batch_start: int` — Index tracking position in the open dataset as batches are processed.
- `X_batch: torch.Tensor` — Feature batch for current open data batch.
- `disc_logits: torch.Tensor` — Discriminator output, shape `(batch_size, 2)`.
- `disc_decisions: torch.Tensor` — `argmax(disc_logits, dim=1)`, shape `(batch_size,)`. Value 0 = familiar, value 1 = unfamiliar. This is the `d^{k,o}_j` from the paper's equation (15).
- `familiar_mask: torch.Tensor` — Boolean mask where `disc_decisions == 0` (the sample is familiar).
- `clf_logits: torch.Tensor` — Classifier output for the entire batch, shape `(batch_size, num_classes)`.
- `clf_predictions: torch.Tensor` — `argmax(clf_logits, dim=1)`, shape `(batch_size,)`. The hard label (predicted class).
- `familiar_indices: np.ndarray` — Absolute indices into `hard_labels` for the familiar samples in the current batch, used to place predicted labels at the right positions.

**Note:** Both discriminator and classifier are run with `torch.no_grad()` for efficiency.

---

### 6.8 `run_distillation(self, global_labels: np.ndarray) -> float`

**Purpose:** Implements Step 5 of the SSFL algorithm — "Distillation." The client uses the global labels broadcast by the server as supervision signal to further train its classifier on the open data. The open data samples are those for which the server returned a valid global label (not -1). This is the "knowledge distillation" step where the client learns from the collective wisdom of all clients.

**Parameters:**
- `global_labels: np.ndarray` — Shape `(N_open,)` array of globally voted labels received from the server. Values are integers 0–10 (valid class) or -1 (no consensus reached). Only samples with valid labels are used for training.

**Returns:** `float` — Average distillation training loss over all epochs.

**Internal Variables:**
- `valid_mask: np.ndarray` — Boolean mask of shape `(N_open,)`. `True` where `global_labels != -1`. Selects samples that have a valid global label.
- `X_distill: np.ndarray` — Subset of `self.X_open` where `valid_mask` is `True`. Shape `(N_valid, 23, 5)`.
- `y_distill: np.ndarray` — Corresponding global labels, shape `(N_valid,)`. These act as "teacher labels."
- `distill_loader: DataLoader` — DataLoader wrapping `(X_distill, y_distill)`, `batch_size=self.batch_size`, `shuffle=True`.
- `criterion: nn.CrossEntropyLoss` — Standard cross-entropy loss. The global hard labels are used as ground truth for this training step.
- `epoch: int` — Loop variable for distillation epochs.
- `total_loss: float` — Accumulated loss per epoch.
- `n_batches: int` — Batch counter.
- `X_batch, y_batch: torch.Tensor` — Current mini-batch moved to `self.device`.
- `logits: torch.Tensor` — Classifier output for the distillation batch.
- `loss: torch.Tensor` — Cross-entropy loss against global labels.

**Side effects:** Updates `self.classifier` weights using `self.classifier_optimizer`.

**Round-1 behavior:** In the first round, `global_labels` is all `-1` (no consensus exists yet), so `valid_mask.sum() == 0`. In that case the function **returns 0.0 immediately without constructing a DataLoader or stepping the optimizer** — distillation is a true no-op on round 1. The same short-circuit also kicks in on any later round where voting produced zero valid labels (extremely unlikely but defended against).

---

### 6.9 `get_parameters(self, config: dict) -> List[np.ndarray]`

**Purpose:** Required method by Flower's `NumPyClient` interface. In standard Flower, this returns model weights. In our SSFL system, **we repurpose this method** to return the client's hard label predictions on the open data. This is what gets sent to the server after the client completes Steps 1–3 of SSFL.

**Parameters:**
- `config: dict` — Configuration dict sent by the server. In our system, this may contain the `current_round` number.

**Returns:** `List[np.ndarray]` — A list containing exactly **one element**: the hard labels array of shape `(N_open,)` as a numpy array. Wrapped in a list to conform to Flower's interface.

**Internal Variables:**
- `hard_labels: np.ndarray` — Computed by calling `self.filter_and_predict()`.

---

### 6.10 `set_parameters(self, parameters: List[np.ndarray]) -> None`

**Purpose:** Required method by Flower's `NumPyClient` interface. In standard Flower, this loads model weights from the server. In our SSFL system, **we repurpose this** to receive the global voted labels from the server and store them in `self.global_labels`.

**Parameters:**
- `parameters: List[np.ndarray]` — A list containing one element: the global label array of shape `(N_open,)` sent from the server after the voting step.

**Returns:** `None`. Mutates `self.global_labels`.

**Internal Variables:**
- `self.global_labels: np.ndarray` — Set to `parameters[0]`. This will be used in `run_distillation()`.

---

### 6.11 `fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]`

**Purpose:** The main method called by Flower at the start of each communication round. This is the **orchestrator for all 5 SSFL steps**. It receives global labels from the previous round (in `parameters`), executes Steps 1–5 sequentially, and returns the hard labels for the current round to the server.

**Parameters:**
- `parameters: List[np.ndarray]` — Contains global labels from the previous round. On round 1, this is an array of `-1` (no prior consensus). Passed to `set_parameters()`.
- `config: dict` — Configuration dictionary from the server. Expected keys:
  - `'round'` (int): Current communication round number.
  - `'classifier_epochs'` (int): Optionally override local epoch count.

**Returns:** `Tuple[List[np.ndarray], int, dict]`
- First element: `List[np.ndarray]` — The client's hard label predictions on open data (output of `filter_and_predict()`). Cast to `int64` before transmission. This is what Flower carries to the server for aggregation.
- Second element: `int` — Number of training samples used (set to `N_private` for weighting purposes, though the server doesn't use weights in our voting strategy).
- Third element: `dict` — Metrics dictionary for the round. Contains:
  - `'classifier_loss'` (float): Average classifier training loss.
  - `'discriminator_loss'` (float): Average discriminator training loss.
  - `'distillation_loss'` (float): Average distillation training loss.
  - `'n_familiar'` (int): Number of open samples classified as familiar.
  - `'n_unfamiliar'` (int): Number of open samples classified as unfamiliar.
  - `'client_id'` (int): This client's ID.
  - `'bytes_upload_wire'` (int): **§14 instrumentation.** Actual on-the-wire upload cost of this client's hard-label payload (`hard_labels_int64.nbytes`). Sums to the paper's `C@Dᵒ`-equivalent row once multiplied across clients and rounds.
  - `'bytes_upload_packed'` (int): Paper-fair upload cost (`N_open × 1 byte`), matching Zhao et al.'s uint8-packed accounting in their Table IV.
  - `'confidence_threshold'` (float): The adaptive median threshold θ used this round; averaged across clients by the strategy to produce `avg_confidence_threshold` (useful for diagnosing drift in the familiar/unfamiliar split).
  - `'fit_wall_clock_sec'` (float): End-to-end time spent inside `fit()`. The server aggregates these into `avg_client_fit_sec` and `max_client_fit_sec` so we can quantify straggler effects.

**Internal Variables:**
- `self.current_round: int` — Incremented at the start.
- `current_global_labels: np.ndarray` — Received via `set_parameters(parameters)`. Global labels from previous round.
- `clf_loss: float` — Return value of `train_classifier()`.
- `confidence_scores: np.ndarray` — Return value of `compute_confidence_scores()`.
- `threshold: float` — Return value of `compute_confidence_threshold(confidence_scores)`.
- `X_disc, y_disc: np.ndarray` — Return values of `build_discriminator_dataset(confidence_scores, threshold)`.
- `disc_loss: float` — Return value of `train_discriminator(X_disc, y_disc)`.
- `hard_labels: np.ndarray` — Return value of `filter_and_predict()`. This is the key upload.
- `distill_loss: float` — Return value of `run_distillation(current_global_labels)`. On round 1, `current_global_labels` is all `-1`, so `run_distillation` short-circuits to `0.0` without stepping the optimizer (see 6.8 "Round-1 behavior").
- `n_familiar: int` — Count of non-(-1) values in `hard_labels`.
- `n_unfamiliar: int` — `N_open - n_familiar`.
- `metrics: dict` — Assembled from the above variables and returned.

---

### 6.12 `evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]`

**Purpose:** Called by Flower periodically to evaluate the client's local classifier on a held-out validation portion of its private data. Reports local accuracy and loss to the server so training progress can be monitored.

**Parameters:**
- `parameters: List[np.ndarray]` — Ignored in our system (no model parameters are shared).
- `config: dict` — Configuration from the server. May contain `'val_split'` (float).

**Returns:** `Tuple[float, int, dict]`
- First element: `float` — Validation loss.
- Second element: `int` — Number of validation samples evaluated.
- Third element: `dict` — Metrics, containing `'accuracy'` (float) and `'client_id'` (int).

**Internal Variables:**
- `val_size: int` — Number of samples to use for validation. Set to 20% of `N_private`.
- `X_val, y_val: np.ndarray` — Last `val_size` samples of private data (not shuffled during evaluation).
- `val_loader: DataLoader` — DataLoader for `(X_val, y_val)`, no shuffle.
- `total_loss: float` — Accumulated cross-entropy loss.
- `correct: int` — Count of correctly classified samples.
- `n_samples: int` — Total number of samples evaluated.
- `criterion: nn.CrossEntropyLoss` — Loss function.
- `accuracy: float` — `correct / n_samples`.

---

## 7. Module: `strategy.py`

**Purpose:** Defines the `SSFLStrategy` class, a custom Flower server strategy. Instead of averaging model weights (as in standard FedAvg), this strategy collects hard label arrays from all clients and applies a **majority voting mechanism** to produce a single global label array. This global label array is then broadcast to all clients for the next distillation step.

---

### 7.1 Class: `SSFLStrategy(flwr.server.strategy.Strategy)`

**Constructor: `__init__(self, num_clients, num_classes, n_open_samples, min_fit_clients, min_available_clients, eval_fn, classifier_epochs=5, charge_open_dataset_round=1)`**

**Purpose:** Initializes the strategy with voting configuration and server-side state. Sets up storage for tracking global labels, per-round metrics, and a `CommCostLedger` (§14.3) that records uploaded / broadcast / open-dataset bytes on every call to `aggregate_fit`.

**Parameters:**
- `num_clients: int` — Total number of clients (27 for Scenario 1). Used to configure minimum participation.
- `num_classes: int` — Number of traffic classes (11). Used to size the voting buckets.
- `n_open_samples: int` — Total number of open dataset samples (`N_open`). Used to allocate the global labels array.
- `min_fit_clients: int` — Minimum number of clients that must participate in each round. Set equal to `num_clients` (all clients must participate).
- `min_available_clients: int` — Minimum number of clients that must be available before a round starts. Set equal to `num_clients`.
- `eval_fn: Callable` — Optional function called after each round to evaluate the server-side global model on the test set. Signature: `eval_fn(round_num: int, global_labels: np.ndarray, X_open: np.ndarray) -> dict`. Expected return keys listed in §8.1 — they populate `server_eval_*` fields in `round_metrics`.
- `classifier_epochs: int` — Number of classifier epochs to request from clients in every round's `FitIns` config payload. Defaults to `config.CLASSIFIER_EPOCHS`.
- `charge_open_dataset_round: int` — The round at which the one-shot `C@Dᵒ` cost is charged in the ledger (default `1`). Set to `0` to disable this term if you want to exclude distribution from cumulative MB totals (useful for within-training comparison).

**Internal Attributes:**
- `self.num_clients: int` — Stored total client count.
- `self.num_classes: int` — Stored class count.
- `self.n_open_samples: int` — Stored open dataset size.
- `self.min_fit_clients: int` — Stored minimum fit participation.
- `self.min_available_clients: int` — Stored minimum availability.
- `self.eval_fn: Callable` — Stored evaluation function.
- `self.classifier_epochs: int` — Stored classifier-epoch count passed to each FitIns config.
- `self.charge_open_dataset_round: int` — Stored one-shot charging round.
- `self.global_labels: np.ndarray` — The current global label array of shape `(N_open,)`. Initialized to all `-1` (no labels known yet). Updated by `vote_mechanism()` after each round.
- `self.round_metrics: List[dict]` — Accumulates per-round metric summaries for logging and later analysis. Initially empty. Serialized as `metrics/per_round.json` and flattened into `per_round.csv` on server shutdown.
- `self.comm_cost_ledger: CommCostLedger` — Running accumulator of per-round byte counts (see §14.3). Queried by `metrics.build_summary_report` to compute `C@50`, `C@75`, and `C@Top-Acc`.

---

### 7.2 `initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]`

**Purpose:** Called by Flower before training begins to provide initial parameters. In our system, we send the initial global labels (all -1) to all clients so they receive a valid array in round 1.

**Parameters:**
- `client_manager: ClientManager` — Flower's client manager (not used directly here).

**Returns:** `flwr.common.Parameters` — Flower's parameter container wrapping the initial `self.global_labels` array.

**Internal Variables:**
- `initial_labels: np.ndarray` — Copy of `self.global_labels` (all -1 array).
- `params: flwr.common.Parameters` — Created via `flwr.common.ndarrays_to_parameters([initial_labels])`.

---

### 7.3 `configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]`

**Purpose:** Called by Flower at the start of each round to tell clients what to do. In our system, we send the current global labels (from the previous round's voting) to every available client. This is the mechanism by which voted labels reach clients for distillation.

**Parameters:**
- `server_round: int` — The current round number (1-indexed).
- `parameters: flwr.common.Parameters` — The parameters from the last `aggregate_fit` call (our global labels). Ignored here — we always send `self.global_labels`.
- `client_manager: ClientManager` — Used to sample clients for this round.

**Returns:** `List[Tuple[ClientProxy, FitIns]]` — One entry per client to be trained this round. Each `FitIns` contains the global labels and a configuration dict with `'round'` number.

**Internal Variables:**
- `config: dict` — Configuration sent to each client. Contains `{'round': server_round, 'classifier_epochs': 5}`.
- `global_labels_params: flwr.common.Parameters` — The current `self.global_labels` array wrapped as Flower Parameters.
- `fit_ins: flwr.common.FitIns` — Instruction object containing `global_labels_params` and `config`.
- `clients: List[ClientProxy]` — All available clients sampled from `client_manager`.
- `fit_configs: List[Tuple]` — The assembled list of `(client, fit_ins)` pairs returned to Flower.

---

### 7.4 `aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List) -> Tuple[Optional[Parameters], dict]`

**Purpose:** Called after all clients return from `fit()`. This is the **core aggregation step** — it collects each client's hard label array and calls `vote_mechanism()` to produce the new global labels. The resulting global labels are stored in `self.global_labels` for the next round.

**Parameters:**
- `server_round: int` — Current round number.
- `results: List[Tuple[ClientProxy, FitRes]]` — List of `(client_proxy, fit_result)` pairs from all participating clients. Each `FitRes` contains the client's hard label array and reported metrics.
- `failures: List` — List of clients that failed to respond (empty if all clients succeed).

**Returns:** `Tuple[Optional[flwr.common.Parameters], dict]`
- First element: Flower `Parameters` object wrapping the new `self.global_labels`. This value is passed to `configure_fit()` next round.
- Second element: `dict` of aggregated metrics for this round.

**Internal Variables:**
- `all_client_labels: List[np.ndarray]` — Collected list of hard label arrays. One array per client, each of shape `(N_open,)`. Built by extracting `parameters_to_ndarrays(fit_res.parameters)[0]` from each `FitRes`.
- `client_metrics: List[dict]` — Collected per-client metrics from `fit_res.metrics`.
- `new_global_labels: np.ndarray` — Result of `self.vote_mechanism(all_client_labels)`.
- `agg_metrics: dict` — Aggregated summary metrics for this round (average losses, total familiar samples, etc.).

**§14 round_metrics schema (aggregated keys written to `self.round_metrics`):**
- `round`, `valid_global_labels`, `total_familiar`, `total_unfamiliar`.
- `avg_classifier_loss`, `avg_discriminator_loss`, `avg_distillation_loss` — means over participating clients.
- `avg_client_fit_sec`, `max_client_fit_sec` — wall-clock diagnostics; spotlight stragglers.
- `server_vote_sec` — time spent inside `vote_mechanism` (pure numpy; small but tracked for big-N scaling studies).
- `server_eval_sec` — time spent inside the optional `eval_fn`.
- `round_total_sec` — full `aggregate_fit` wall-clock.
- `avg_confidence_threshold` — averaged median-θ across clients.
- `bytes_upload_wire_total`, `bytes_upload_packed_total` — summed across clients this round.
- `bytes_broadcast_wire_total`, `bytes_broadcast_packed_total` — one global-label array × `len(results)` (fan-out).
- `bytes_open_dataset_this_round` — non-zero only at `charge_open_dataset_round` (default 1); equals `N_open × N_features × 4 × num_clients` in wire mode.
- `cumulative_mb_wire`, `cumulative_mb_packed` — running totals via `self.comm_cost_ledger.cumulative_mb_at(round)`.
- `server_eval_*` — every key returned by `eval_fn` is prefixed `server_eval_` and merged in (e.g. `server_eval_accuracy`, `server_eval_f1_macro`, `server_eval_confusion_matrix`, per-class lists). Absent when the closure is stubbed.

**Side effects:** Updates `self.global_labels` with the voted result. Appends one entry to `self.round_metrics`. Appends one `RoundCommCost` entry to `self.comm_cost_ledger`.

---

### 7.5 `vote_mechanism(self, all_client_labels: List[np.ndarray]) -> np.ndarray`

**Purpose:** The **central algorithm of the SSFL server**. For each open sample position `j`, collects all non-(-1) predictions from all clients, counts votes per class, and assigns the majority-voted class. If no client provided a label for a sample (all predicted unfamiliar), the global label remains `-1`.

**Parameters:**
- `all_client_labels: List[np.ndarray]` — List of K=27 arrays, each of shape `(N_open,)`. Value at index `j` is either a class integer (0–10) or `-1` (unfamiliar/abstained).

**Returns:** `np.ndarray` — Global label array of shape `(N_open,)` with voted labels. Each value is either a valid class integer 0–10 or `-1` if no votes were cast.

**Internal Variables:**
- `global_labels: np.ndarray` — Output array, initialized to all `-1`, shape `(N_open,)`. Gets filled in as votes are counted.
- `voting_sets: np.ndarray` — A 2D array of shape `(N_open, num_classes)` used as a vote accumulator. `voting_sets[j, c]` counts how many clients predicted class `c` for open sample `j`. Initialized to all zeros. Corresponds to the `V_{j,l}` sets in the paper's equation (17).
- `client_labels: np.ndarray` — Loop variable, one client's full label array during iteration.
- `sample_idx: int` — Index into the open dataset (0 to N_open-1) during the voting loop.
- `predicted_class: int` — The predicted class from one client for one sample. Skip if `-1`.
- `vote_counts: np.ndarray` — For one sample, the row `voting_sets[j]` of shape `(num_classes,)`. The argmax of this gives the winning class.
- `total_votes: int` — Sum of `vote_counts` for sample `j`. If 0, no client provided a label → stays -1.
- `winning_class: int` — `argmax(vote_counts)` — the class with the most votes for sample `j`.

**Complexity note:** An efficient implementation processes all clients in a vectorized manner by building `voting_sets` using numpy indexing rather than a nested Python loop.

---

### 7.6 `configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]`

**Purpose:** Called by Flower to configure the evaluation step after each training round. Tells clients to evaluate their local classifier on their validation data.

**Parameters:**
- `server_round: int` — Current round number. Evaluation may be skipped every N rounds to save time.
- `parameters: Parameters` — Current global parameters (our global labels).
- `client_manager: ClientManager` — Used to sample clients for evaluation.

**Returns:** `List[Tuple[ClientProxy, EvaluateIns]]` — Evaluation instructions per client.

**Internal Variables:**
- `eval_config: dict` — Configuration for clients, containing `{'round': server_round}`.
- `evaluate_ins: EvaluateIns` — Instruction object.
- `clients: List[ClientProxy]` — Subset of clients to evaluate (can be a fraction for efficiency).

---

### 7.7 `aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List) -> Tuple[Optional[float], dict]`

**Purpose:** Aggregates evaluation results from all participating clients. Computes weighted average accuracy and loss across clients, weighted by their number of validation samples.

**Parameters:**
- `server_round: int` — Current round number.
- `results: List[Tuple[ClientProxy, EvaluateRes]]` — List of evaluation results from clients.
- `failures: List` — Clients that failed evaluation.

**Returns:** `Tuple[Optional[float], dict]`
- Aggregated loss (weighted average).
- Dict with `'accuracy'` (weighted average), `'round'` (int).

**Internal Variables:**
- `total_samples: int` — Sum of all clients' validation sample counts.
- `weighted_loss: float` — Weighted sum of losses.
- `weighted_accuracy: float` — Weighted sum of accuracies.
- `loss_agg: float` — `weighted_loss / total_samples`.
- `accuracy_agg: float` — `weighted_accuracy / total_samples`.

---

## 8. Module: `server.py`

**Purpose:** Handles the server-side setup and launches the Flower server. Creates the strategy, optionally loads the test dataset for server-side evaluation, and starts the Flower server loop.

---

### 8.1 `build_eval_fn(X_test: np.ndarray, y_test: np.ndarray, num_classes: int, device: torch.device, server_eval_epochs: int = config.SERVER_EVAL_EPOCHS) -> Callable`

**Purpose:** Creates a closure (a function-returning-function) that the strategy can call after each round to evaluate a server-side global model on the held-out test set. The global model is re-trained from scratch using the current voted labels on the open dataset.

**Parameters:**
- `X_test: np.ndarray` — Test feature array, shape `(N_test, 23, 5)`.
- `y_test: np.ndarray` — Test label array, shape `(N_test,)`.
- `num_classes: int` — Number of traffic classes.
- `device: torch.device` — Compute device.
- `server_eval_epochs: int` — Epochs used when re-training the temporary server classifier on `(X_open[valid], global_labels[valid])`. Default `config.SERVER_EVAL_EPOCHS == 10`.

**Returns:** `Callable` — A function with signature `eval_fn(server_round, global_labels, X_open) -> dict`.

**Expected return schema (what each call produces — mirrors `metrics.compute_classification_metrics` exactly, so every key passes through the strategy prefixed with `server_eval_`):**
- `accuracy: float`
- `f1_macro: float`, `f1_weighted: float`
- `precision_macro: float`, `precision_weighted: float`
- `recall_macro: float`, `recall_weighted: float`
- `f1_per_class: List[float]` (length `num_classes`)
- `precision_per_class: List[float]`, `recall_per_class: List[float]`, `support_per_class: List[int]`
- `confusion_matrix: List[List[int]]` — `num_classes × num_classes`; consumed by `save_confusion_matrix_json` at shutdown to produce Fig. 3.
- `class_names: List[str]` — Human-readable class labels aligned with `config.GLOBAL_ID_TO_CLASS_NAME`.

**Internal Variables inside the returned closure (post-CNN implementation):**
- `valid_mask: np.ndarray` — `global_labels != -1`; the subset of open samples with a voted consensus.
- `server_model: TrafficCNN` — Fresh classifier (no carry-over between rounds, per paper §II-B).
- `train_loader: DataLoader` — For `(X_open[valid_mask], global_labels[valid_mask])`.
- `test_loader: DataLoader` — For `(X_test, y_test)`, no shuffle.
- `y_pred: np.ndarray` — Argmax of classifier logits on `X_test`.
- Return is `compute_classification_metrics(y_test, y_pred, num_classes)`.

**Stub behavior:** Raises `NotImplementedError` until `model.py` ships. The strategy catches this specifically, logs at DEBUG, and continues — so pre-CNN runs exercise voting + comm-cost accounting end-to-end but emit empty `server_eval_*` fields.

---

### 8.2 `start_server(server_address: str, strategy: SSFLStrategy, num_rounds: int) -> flwr.server.History`

**Purpose:** Launches the Flower server with the given strategy and runs it for `num_rounds` communication rounds.

**Parameters:**
- `server_address: str` — The host:port string for the gRPC server (e.g., `"0.0.0.0:8080"`).
- `strategy: SSFLStrategy` — The configured SSFL strategy instance.
- `num_rounds: int` — Total number of communication rounds (T in the paper). Paper uses ~150 rounds for convergence.

**Returns:** `flwr.server.History` — Flower's history object containing per-round metrics from all rounds. Saved to disk for later analysis.

**Internal Variables:**
- `server_config: flwr.server.ServerConfig` — Wraps `num_rounds` into Flower's config format.
- `history: flwr.server.History` — The returned object from `flwr.server.start_server()`.

---

## 9. Module: `main.py`

**Purpose:** Entry point for both server and client processes. Parses command-line arguments to determine whether the current process should run as a server or as one specific client. This design allows running multiple client processes independently (simulating separate IoT devices).

---

### 9.1 `parse_arguments() -> argparse.Namespace`

**Purpose:** Defines and parses all command-line arguments needed to configure a server or client process.

**Parameters:** None (reads from `sys.argv`).

**Returns:** `argparse.Namespace` — Parsed argument object with the following attributes:

- `args.mode: str` — Either `'server'` or `'client'`. Determines which role this process plays.
- `args.client_id: int` — (Client mode only) Global client ID (0–26). Determines which partition to load.
- `args.server_address: str` — Host:port for the Flower gRPC connection. Default `"127.0.0.1:8080"`.
- `args.num_rounds: int` — (Server mode only) Number of training rounds. Default `150`.
- `args.num_clients: int` — (Server mode only) Total number of clients. Default `27`.
- `args.num_classes: int` — Number of traffic classes. Default `11`.
- `args.learning_rate: float` — Adam optimizer learning rate. Default `0.0001`.
- `args.batch_size: int` — Mini-batch size. Default `100`.
- `args.classifier_epochs: int` — Local classifier training epochs per round. Default `5`.
- `args.discriminator_epochs: int` — Discriminator training epochs per round. Default `5`.
- `args.partition_dir: str` — Directory where pre-partitioned data files are stored.
- `args.data_dir: str` — Raw data directory (used during data preparation only).
- `args.device: str` — `'cuda'` or `'cpu'`. Default `'cpu'`.
- `args.seed: int` — Random seed for reproducibility. Default `42`.
- `args.metrics_dir: str` — (Server mode) Directory for `per_round.{json,csv}`, `summary.json`, `confusion_matrix_final.json`. Default `config.METRICS_DIR == "metrics"`.
- `args.logs_dir: str` — (Server mode) Directory for the Flower `history.json`. Default `config.LOGS_DIR == "logs"`.
- `args.target_accs: str` — (Server mode) Comma-separated target accuracies for C@x (default `"0.50,0.75"`). Parsed into a `Tuple[float, ...]` in `run_server`.
- `args.snapshot_rounds: str` — (Server mode) Comma-separated rounds at which Top-1 accuracy is snapshotted for Table III (default `"10,50,100,150,200"`). Parsed into a `Tuple[int, ...]`.

**Internal Variables:**
- `parser: argparse.ArgumentParser` — The argument parser object.

---

### 9.2 `setup_environment(seed: int) -> None`

**Purpose:** Sets all random seeds for reproducibility across NumPy, PyTorch, and Python's random module. Also configures logging format.

**Parameters:**
- `seed: int` — The random seed value.

**Returns:** `None`. Side effects: sets seeds globally.

**Internal Variables:**
- None. Calls `np.random.seed(seed)`, `torch.manual_seed(seed)`, `random.seed(seed)`.

---

### 9.3 `run_server(args: argparse.Namespace) -> None`

**Purpose:** Constructs all server-side objects, starts the Flower server, and **on shutdown** persists the full metrics bundle (§14.5) so downstream plotting / report-writing never touches live training code.

**Parameters:**
- `args: argparse.Namespace` — Parsed command-line arguments.

**Returns:** `None`.

**Internal Variables:**
- `device: torch.device` — Resolved from `args.device`.
- `X_test, y_test: np.ndarray` — Loaded from `args.partition_dir` via `load_test_data`.
- `X_open: np.ndarray` — Loaded via `load_open_data`; only `X_open.shape[0]` is consumed here (server doesn't hold the data in memory).
- `n_open_samples: int` — `int(X_open.shape[0])`. Sizes the global labels array.
- `target_accs: Tuple[float, ...]`, `snapshot_rounds: Tuple[int, ...]` — Parsed from `args.target_accs` / `args.snapshot_rounds` via `_parse_float_tuple` / `_parse_int_tuple`.
- `eval_fn: Callable` — Created by `build_eval_fn(X_test, y_test, args.num_classes, device)`.
- `strategy: SSFLStrategy` — Constructed with `eval_fn`, `classifier_epochs`, and the default `charge_open_dataset_round=1`.
- `history: flwr.server.History` — Returned by `start_server()`. Written as JSON to `{args.logs_dir}/history.json` alongside a copy of `strategy.round_metrics`.

**§14.5 shutdown artefacts written under `args.metrics_dir`:**
- `per_round.json` — `metrics.save_metrics_json(strategy.round_metrics, ...)`. Keeps every list-valued and per-class field so the confusion matrix per round is preserved.
- `per_round.csv` — `metrics.save_metrics_csv(...)`. Scalar-only, one row per round. Ideal for quick pandas plotting.
- `summary.json` — `metrics.build_summary_report(strategy.round_metrics, strategy.comm_cost_ledger, target_accs, snapshot_rounds)` + the full `comm_cost_ledger.to_dict()` blob appended as `"comm_cost_ledger"`. Contains `top_acc`, `top_acc_round`, `c_at_top_acc_wire_mb`, `c_at_top_acc_packed_mb`, `c_at_open_dataset_bytes/mb`, `accuracy_snapshots`, `c_at_accuracy_wire_mb`, `c_at_accuracy_packed_mb`, `final_round_cumulative_{wire,packed}_mb`.
- `confusion_matrix_final.json` — Extracted from the last `server_eval_confusion_matrix` entry in `round_metrics` + the matching per-class lists and `class_names`. Skipped (with log note) if `eval_fn` was stubbed for the whole run.

---

### 9.4 `run_client(args: argparse.Namespace) -> None`

**Purpose:** Constructs the client object for the specified `client_id` and starts the Flower client. Loads the client's private partition and the shared open data, builds both models, and calls `flwr.client.start_client(server_address=..., client=ssfl_client.to_client())`. Note: `flwr.client.start_numpy_client()` is deprecated in Flower ≥ 1.4; we therefore call `start_client(...)` and convert our `NumPyClient` instance using its `.to_client()` method.

**Parameters:**
- `args: argparse.Namespace` — Parsed command-line arguments.

**Returns:** `None`.

**Internal Variables:**
- `device: torch.device` — Resolved from `args.device`.
- `X_private, y_private: np.ndarray` — Loaded via `load_client_partition(args.client_id, args.partition_dir)`.
- `X_open: np.ndarray` — Loaded via `load_open_data(args.partition_dir)`. Same for all clients.
- `client: SSFLClient` — Constructed with all loaded data and config arguments.

---

### 9.5 `main() -> None`

**Purpose:** Entry point. Calls `parse_arguments()`, `setup_environment()`, then either `run_server()` or `run_client()` depending on `args.mode`.

**Parameters:** None.

**Returns:** `None`.

**Internal Variables:**
- `args: argparse.Namespace` — Parsed arguments.

---

## 10. Module: `utils.py`

**Purpose:** Shared helper utilities used across multiple modules — thin I/O wrappers, logging setup, feature-name lookup, and a backward-compatible `compute_metrics` **facade** over the new `metrics.py` (see §14). All authoritative classification math and communication-cost accounting have moved to `metrics.py`; `utils.py` is only the side-effect layer.

---

### 10.1 `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict` *(facade)*

**Purpose:** Backward-compatible facade over `metrics.compute_classification_metrics` (§14.1). Exists solely so older call sites and unit tests that imported `utils.compute_metrics` keep working; new code should import from `metrics` directly.

**Parameters:**
- `y_true: np.ndarray` — Ground truth integer class labels.
- `y_pred: np.ndarray` — Predicted integer class labels.
- `num_classes: int` — Total number of classes.

**Returns:** `dict` — same schema as `metrics.compute_classification_metrics`, **except** `'confusion_matrix'` is converted back to an `np.ndarray` (shape `(num_classes, num_classes)`) rather than a list-of-lists. This preserves the legacy contract. All richer keys (`f1_weighted`, per-class lists, `class_names`) flow through unchanged.

**Internal:**
- Delegates to `metrics.compute_classification_metrics(y_true, y_pred, num_classes)`; converts the list-of-lists confusion matrix to `np.asarray(..., dtype=np.int64)` before returning.

---

### 10.2 `save_round_metrics(round_num: int, metrics: dict, output_path: str) -> None`

**Purpose:** Appends per-round metrics to a JSON log file on disk. Used by the strategy as an optional "live" writer during long runs so a crash doesn't lose intermediate data.

**Parameters:**
- `round_num: int` — The communication round number.
- `metrics: dict` — Metrics dict for this round (may contain numpy scalars/arrays; these are sanitized by `_json_safe`).
- `output_path: str` — Path to the JSON log file (appended to, not overwritten).

**Returns:** `None`.

**Note:** The canonical end-of-run dumps live in `metrics.save_metrics_json` / `save_metrics_csv` / `save_summary_json` (§14.5). `save_round_metrics` is the *streaming* variant for progress tracking.

---

### 10.3 `get_feature_column_names() -> List[str]`

**Purpose:** Thin accessor that returns `config.FEATURE_COLUMN_NAMES`. Centralising this lookup in `utils` keeps the call-site API stable even when `config`'s internals change.

**Parameters:** None.

**Returns:** `List[str]` — The 115 N-BaIoT feature column names, ordered so flat index `j*23 + k` maps to (feature k, time window j) after `reshape_sample_to_2d`.

---

### 10.4 `setup_logging(level: int = logging.INFO) -> None`

**Purpose:** Configures the root logger with a consistent timestamped format across the server process and all 27 client processes, so parsing aggregated logs is trivial.

**Parameters:**
- `level: int` — Python logging level (default `logging.INFO`).

**Returns:** `None`.

---

## 10A. Module: `config.py`

**Purpose:** Single source of truth for all hyperparameters, filesystem paths, dataset constants, and identifier mappings used across the system. Every other module imports from here instead of hard-coding numbers or strings. `main.parse_arguments()` wires its `--flag` defaults to these constants so the CLI and the code cannot drift apart.

**Contents (module-level constants — no functions, no classes):**

- `NUM_DEVICES: int` — `9`. Number of IoT devices in N-BaIoT.
- `K_DI: int` — `3`. Number of clients per device in Scenario 1.
- `NUM_CLIENTS: int` — `NUM_DEVICES * K_DI == 27`. Total Scenario 1 client count.
- `NUM_CLASSES: int` — `11`. Size of the **global label space** (1 benign + 10 attack families, union across all devices).
- `N_FEATURES: int` — `115`. Number of raw features per N-BaIoT sample.
- `N_TIME_WINDOWS: int` — `5`. Time windows (100 ms, 500 ms, 1.5 s, 10 s, 1 min).
- `FEATURES_PER_WINDOW: int` — `23`. Satisfies `N_TIME_WINDOWS * FEATURES_PER_WINDOW == N_FEATURES`.
- `INPUT_SHAPE: Tuple[int, int]` — `(FEATURES_PER_WINDOW, N_TIME_WINDOWS) == (23, 5)`. The shape every sample is reshaped to.
- `SAMPLES_PER_CLASS: int` — `1000`. Mini-N-BaIoT sampling budget per `(device, class)`.
- `PRIVATE_RATIO: float` — `0.70`.
- `OPEN_RATIO: float` — `0.10`.
- `TEST_RATIO: float` — `0.20`. (`PRIVATE_RATIO + OPEN_RATIO + TEST_RATIO == 1.0`.)
- `LEARNING_RATE: float` — `1e-4`.
- `BATCH_SIZE: int` — `100`.
- `CLASSIFIER_EPOCHS: int` — `5`.
- `DISCRIMINATOR_EPOCHS: int` — `5`.
- `DISTILLATION_EPOCHS: int` — `5`.
- `NUM_ROUNDS: int` — `150`. Paper's approximate convergence point.
- `RANDOM_SEED: int` — `42`.
- `DEFAULT_SERVER_ADDRESS: str` — `"127.0.0.1:8080"`.
- `DATA_DIR: str`, `RAW_DIR: str`, `PROCESSED_DIR: str`, `PARTITION_DIR: str` — Canonical subfolders derived from `DATA_DIR`.
- `METRICS_DIR: str` — `"metrics"`. Where `per_round.{json,csv}`, `summary.json`, and `confusion_matrix_final.json` are written (see §14.5).
- `LOGS_DIR: str` — `"logs"`. Where the Flower-native `history.json` lands.
- `FEATURE_COLUMN_NAMES: List[str]` — The 115 N-BaIoT feature column names, returned by `utils.get_feature_column_names()`. Stored here so every module sees the same ordering.
- `CLASS_NAME_TO_GLOBAL_ID: Dict[str, int]` — Lookup from traffic-category filename stem (e.g., `"benign"`, `"mirai_udp"`, `"gafgyt_combo"`) to its **global** integer label in 0..10. This is the authority that `data_preparation.load_device_csvs` consults when labeling rows, and it is what guarantees that a client never has to know which classes its device is missing — it just never emits labels for them.
- `GLOBAL_ID_TO_CLASS_NAME: Dict[int, str]` — Inverse of the above, used only for logging and confusion-matrix pretty-printing.

**Evaluation-protocol constants (§14):**
- `SNAPSHOT_ROUNDS: Tuple[int, ...]` — `(10, 50, 100, 150, 200)`. Rounds at which Top-1 accuracy is snapshotted to mirror Zhao et al. Table III.
- `TARGET_ACCURACIES: Tuple[float, ...]` — `(0.50, 0.75)`. Accuracy thresholds used to populate the C@50 / C@75 columns of our Table IV analogue.
- `SERVER_EVAL_EPOCHS: int` — `10`. Epochs the server-side `eval_fn` uses when re-training a fresh classifier on `(X_open[valid], global_labels[valid])` before scoring on the held-out test set.

**Byte-accounting constants (§14.2):**
- `BYTES_PER_FLOAT32: int` — `4`.
- `BYTES_PER_INT64: int` — `8`.
- `BYTES_PER_HARD_LABEL_WIRE: int` — `8`. What a single hard label actually costs on the wire when serialized as numpy `int64` through Flower/gRPC.
- `BYTES_PER_HARD_LABEL_PAPER: int` — `1`. Paper-fair accounting (`uint8`); 4 bits would suffice for 12 label values but byte-alignment is the standard comparator.
- `BYTES_PER_OPEN_SAMPLE_FP32: int` — `N_FEATURES * 4 == 460`. Size of a single open-dataset sample in the `float32` representation that's distributed server→client once.
- `BYTES_PER_OPEN_SAMPLE_UINT8: int` — `N_FEATURES == 115`. Paper-fair quantized baseline.
- `ESTIMATED_CNN_PARAM_COUNT: int` — `300_000`. Conservative estimate for the 8-conv + 2-FC classifier; used by `metrics.fl_baseline_upload_bytes` to produce the "FL" row in Table IV **before** `model.py` ships. Once it ships, the live value should be recomputed from `sum(p.numel() for p in classifier.parameters())`.

**No logic lives here.** If a calculation is needed (e.g., "how many shards per device"), it is computed in the module that uses it, consuming these constants as inputs.

---

## 10B. Module: `launch.py` (optional — multi-process orchestration)

**Purpose:** Convenience wrapper that starts one server process and 27 client processes locally, forwarding the right `--mode` and `--client_id` to each. Only used when running the system on a single machine as separate OS processes. Not needed if using Flower's in-process simulation (see section 10C).

**Behavior:** spawns one `python main.py --mode server` subprocess, waits briefly for the gRPC listener, then spawns 27 `python main.py --mode client --client_id {i}` subprocesses with `i` in `0..26`. Uses `subprocess.Popen` with per-process log redirection to `logs/client_{i}.log` so debugging one client doesn't require parsing a single mixed log. Propagates `SIGINT` to all children on Ctrl-C.

---

## 10C. Run Modes: Separate Processes vs. Flower Simulation

Scenario 1 needs 27 clients to participate concurrently. The plan supports two orthogonal ways to achieve that, chosen at CLI level and not baked into the module code:

**Mode A — Separate OS processes (default, matches the CLI as described in section 9).** One server process plus 27 client processes, each started via `python main.py --mode {server|client} --client_id {i}`. This is the mode that exercises Flower's real gRPC transport, so it's the most faithful test of the Flower contract described in section 13. `launch.py` (section 10B) is the recommended helper. Each client holds its private partition in its own address space; only hard-label arrays and the gRPC config dict cross process boundaries. Scales well up to ~30 clients on a workstation with 16+ GB RAM.

**Mode B — Flower simulation (`flwr.simulation.start_simulation`) via Ray.** All 27 clients run inside a single Python process, each in its own Ray actor. Same `SSFLClient` class, same `SSFLStrategy` — no code changes — just invoked through a different entry point that supplies a `client_fn(cid) -> SSFLClient` factory. Mode B is the recommended mode for **large sweeps**: faster startup, no gRPC overhead, and trivial to parameterize across `num_rounds`, seeds, or hyperparameters. It is **not** a substitute for Mode A during contract validation, because it bypasses the wire format entirely.

A future `main.py` flag `--run_mode {processes,simulation}` can expose the toggle; for the first infrastructure pass, only Mode A is implemented and Mode B is a documented extension point.

---

## 11. Key Data Structures Reference

This section summarizes all major data structures that flow through the system.

| Variable Name | Type | Shape / Contents | Where Created | Purpose |
|---|---|---|---|---|
| `X_private` | `np.ndarray` | `(N_client, 23, 5)` float32 | `data_preparation.py` | Client's private labeled features |
| `y_private` | `np.ndarray` | `(N_client,)` int64 | `data_preparation.py` | Client's private class labels |
| `X_open` | `np.ndarray` | `(N_open, 23, 5)` float32 | `data_preparation.py` | Shared unlabeled open data features |
| `confidence_scores` | `np.ndarray` | `(N_open,)` float32 | `client.compute_confidence_scores()` | Per-sample classifier confidence |
| `threshold` | `float` | scalar | `client.compute_confidence_threshold()` | Median confidence, used for familiar/unfamiliar split |
| `X_disc` | `np.ndarray` | `(N_disc, 23, 5)` float32 | `client.build_discriminator_dataset()` | Discriminator training features |
| `y_disc` | `np.ndarray` | `(N_disc,)` int64, values 0 or 1 | `client.build_discriminator_dataset()` | Discriminator training labels |
| `hard_labels` | `np.ndarray` | `(N_open,)` int64, values -1 to 10 | `client.filter_and_predict()` | Client's predictions uploaded to server |
| `all_client_labels` | `List[np.ndarray]` | List of K arrays, each `(N_open,)` | `strategy.aggregate_fit()` | All clients' predictions before voting |
| `voting_sets` | `np.ndarray` | `(N_open, num_classes)` int | `strategy.vote_mechanism()` | Vote accumulator per sample per class |
| `global_labels` | `np.ndarray` | `(N_open,)` int64, values -1 to 10 | `strategy.vote_mechanism()` | Voted global labels broadcast to all clients |
| `self.global_labels` | `np.ndarray` | `(N_open,)` int64 | `client.set_parameters()` | Client's stored copy of global labels |
| `round_metrics` | `List[dict]` | List grows each round | `strategy.aggregate_fit()` | Per-round metrics log (§14 schema) |
| `all_partitions` | `Dict[int, pd.DataFrame]` | 27 entries | `data_preparation.build_all_client_partitions()` | Maps client ID to its data |
| `RoundCommCost` | `dataclass` | One record per round | `strategy.aggregate_fit()` (via `metrics.py`) | Upload / broadcast / open-dataset bytes in both wire and packed accounting; `total_wire` / `total_packed` properties |
| `CommCostLedger` | `class` | `List[RoundCommCost]` + helpers | `strategy.__init__` | Running byte total; exposes `cumulative_mb_at(round, packed=bool)` and `cumulative_series(packed=bool)` for the C@x column of Table IV |
| `summary_report` | `dict` | 1 record per run | `metrics.build_summary_report()` | Flat schema of every Table II-IV cell; written to `metrics/summary.json` at shutdown |

---

## 12. Full Training Round Flow

The following describes exactly what happens during one complete communication round `t`, tracing execution from server through clients and back.

### Pre-Round (Server)
1. `strategy.configure_fit(t, params, client_manager)` is called by Flower.
2. `self.global_labels` (shape `(N_open,)`) is packaged as Flower `Parameters`.
3. A `FitIns` is constructed with global labels + config `{'round': t}`.
4. The same `FitIns` is sent to all 27 clients.

### Client Execution (All 27 clients in parallel)
5. Flower calls `client.fit(parameters, config)` on each client.
6. `set_parameters(parameters)` → `self.global_labels` updated with server's voted labels.
7. **Step 1:** `train_classifier()` → classifier trained on private labeled data for 5 epochs.
8. **Step 1 (cont):** `compute_confidence_scores()` → confidence scores for all N_open samples.
9. **Step 1 (cont):** `compute_confidence_threshold()` → compute median threshold `θ`.
10. **Step 2:** `build_discriminator_dataset()` → assemble familiar/unfamiliar training set.
11. **Step 2 (cont):** `train_discriminator()` → discriminator trained on familiar/unfamiliar data.
12. **Step 3:** `filter_and_predict()` → for each open sample, discriminator decides familiar/unfamiliar, classifier provides hard label for familiar samples. Result: `hard_labels` array.
13. **Step 5:** `run_distillation(self.global_labels)` → classifier fine-tuned on open samples that have a valid global label (not -1).
14. `fit()` returns `([hard_labels], N_private, metrics_dict)`.

### Post-Round (Server)
15. Flower calls `strategy.aggregate_fit(t, results, failures)`.
16. All 27 `hard_labels` arrays are extracted from `results`.
17. `vote_mechanism(all_client_labels)` runs: for each of the N_open samples, votes are tallied across 27 clients → majority class wins → new `global_labels` array produced.
18. `self.global_labels` is updated with the new voted labels.
19. New `global_labels` is returned as Flower `Parameters` (used in next round's `configure_fit`).

### Evaluation (Periodic)
20. Flower calls `strategy.configure_evaluate()` → sends evaluation instruction.
21. Clients call `evaluate()` → measure local classifier accuracy.
22. `strategy.aggregate_evaluate()` → compute weighted average accuracy across clients.

This cycle repeats for T rounds (T ≈ 150 for convergence as observed in the paper).

---

## 13. Flower Communication Contract

### What Travels Over the Wire (Client → Server)

| Content | Type | Shape | Flower Field |
|---|---|---|---|
| Client's hard labels on open data | `np.ndarray` int64 | `(N_open,)` | `FitRes.parameters` |
| Classifier training loss | float | scalar | `FitRes.metrics['classifier_loss']` |
| Discriminator training loss | float | scalar | `FitRes.metrics['discriminator_loss']` |
| Distillation training loss | float | scalar | `FitRes.metrics['distillation_loss']` |
| Number of familiar samples | int | scalar | `FitRes.metrics['n_familiar']` |
| Number of unfamiliar samples | int | scalar | `FitRes.metrics['n_unfamiliar']` |
| Client ID | int | scalar | `FitRes.metrics['client_id']` |
| Upload bytes (wire) | int | scalar | `FitRes.metrics['bytes_upload_wire']` — `hard_labels.nbytes` (§14.2) |
| Upload bytes (packed) | int | scalar | `FitRes.metrics['bytes_upload_packed']` — `N_open × 1 byte` (§14.2) |
| Adaptive confidence θ | float | scalar | `FitRes.metrics['confidence_threshold']` (§14 diagnostic) |
| Client fit wall-clock | float | scalar (sec) | `FitRes.metrics['fit_wall_clock_sec']` (§14 diagnostic) |

### What Travels Over the Wire (Server → Client)

| Content | Type | Shape | Flower Field |
|---|---|---|---|
| Global voted labels from previous round | `np.ndarray` int64 | `(N_open,)` | `FitIns.parameters` |
| Current round number | int | scalar | `FitIns.config['round']` |
| Classifier epochs override | int | scalar | `FitIns.config['classifier_epochs']` |

### Privacy Guarantee

- **Model weights NEVER leave the client.** The classifier and discriminator parameters are never put into `get_parameters()`.
- **Raw private data NEVER leaves the client.** Only the predicted label (a single integer per open sample) is shared.
- **Only hard labels are shared.** A hard label is a class index integer — it carries far less information than a soft label vector or gradient vector.

---

## 14. Metrics & Evaluation Protocol

**Purpose:** This section is the contract for every number that will appear in our capstone report. It guarantees that our reproduction of Zhao et al. (IEEE IoT Journal, 2023) publishes the *same tables* (Table II classification breakdown, Table III Top-1 accuracy across rounds, Table IV communication overhead), *plus* diagnostic metrics we found useful but the paper omits (wall-clock, per-class F1, per-round byte breakdown).

All math lives in `metrics.py`. `utils.py` keeps a backward-compatible facade (§10.1). Side-effect-free + CNN-independent modules are marked **live**; anything labelled **stubbed** raises `NotImplementedError` until `model.py` ships but has a finalized signature.

---

### 14.1 Classification metrics — Table II analogue (**live**)

**Function:** `metrics.compute_classification_metrics(y_true, y_pred, num_classes, class_names=None) -> dict`

**Inputs:** Two 1-D int arrays `y_true`, `y_pred` of equal length, plus `num_classes` (11 for N-BaIoT Mini) and an optional `class_names` override — if omitted, the mapping comes from `config.GLOBAL_ID_TO_CLASS_NAME`.

**Returns (flat dict with per-class lists and 2-D confusion matrix):**

| Key | Type | Meaning |
|---|---|---|
| `accuracy` | float | Overall top-1 accuracy |
| `f1_macro` / `f1_weighted` | float | Macro and support-weighted F1 (paper reports macro; weighted is our addition to catch class-imbalance artefacts) |
| `precision_macro` / `precision_weighted` | float | Macro + weighted precision |
| `recall_macro` / `recall_weighted` | float | Macro + weighted recall |
| `f1_per_class` | `List[float]` | Length `num_classes`; drives the per-class bar chart in our final report |
| `precision_per_class` / `recall_per_class` / `support_per_class` | List | Length `num_classes`; support is `int` |
| `confusion_matrix` | `List[List[int]]` | Shape `num_classes × num_classes`; backs Fig. 3 heatmap |
| `class_names` | `List[str]` | Human-readable labels aligned with `GLOBAL_ID_TO_CLASS_NAME` |

**Rationale:** Zhao et al. report only `(accuracy, precision, recall, F1)`. We return the per-class breakdown as well because (a) the paper's "minor classes" (e.g., `gafgyt_junk`) are likely where SSFL underperforms FL, and (b) per-class F1 is necessary to defend the "no class is starved" claim in §VI of our capstone report.

---

### 14.2 Payload sizing — Table IV raw counts (**live**)

Two byte models are maintained in parallel, because the paper's accounting and our real Flower implementation disagree — both are valid numbers to cite, and we let the final report pick:

| Helper | Formula | Used for |
|---|---|---|
| `payload_bytes_wire(arr)` | `arr.nbytes` (int64 ⇒ 8 B / label) | Real on-the-wire gRPC cost — reflects what actually leaves the socket in our pass-1 implementation |
| `payload_bytes_packed(arr)` | `arr.size × 1` (uint8) | Paper-fair count — Zhao et al. treat a single hard label as 1 byte |
| `open_dataset_distribution_bytes(n_open, n_features, bytes_per_value)` | `n_open × n_features × bytes_per_value` | The one-shot server→client push of `X_open` before round 1 (paper's `C@Dᵒ`) |
| `fl_baseline_upload_bytes(num_params, bytes_per_param)` | `num_params × bytes_per_param` | Per-round, per-client parameter upload of the vanilla-FL baseline. Defaults to `ESTIMATED_CNN_PARAM_COUNT × 4`; should be recomputed from `sum(p.numel() for p in classifier.parameters())` once `model.py` ships |
| `bytes_to_mb(n_bytes)` | `n_bytes / 1_000_000` | Decimal-MB convention used throughout Zhao et al. Table IV |

**Byte-constant defaults (from `config.py` §10A):** `BYTES_PER_FLOAT32=4`, `BYTES_PER_INT64=8`, `BYTES_PER_HARD_LABEL_WIRE=8`, `BYTES_PER_HARD_LABEL_PAPER=1`, `BYTES_PER_OPEN_SAMPLE_FP32=460`, `BYTES_PER_OPEN_SAMPLE_UINT8=115`, `ESTIMATED_CNN_PARAM_COUNT=300_000`.

---

### 14.3 Running ledger — `RoundCommCost` & `CommCostLedger` (**live**)

**`RoundCommCost` dataclass** — one row per communication round:

| Field | Meaning |
|---|---|
| `round` | 1-indexed round number |
| `uploaded_bytes_wire` | Sum across clients of `FitRes.metrics['bytes_upload_wire']` this round |
| `uploaded_bytes_packed` | Sum across clients of `bytes_upload_packed` |
| `downloaded_bytes_wire` | One global-label broadcast × `num_clients` fan-out |
| `downloaded_bytes_packed` | Same, under the packed accounting |
| `open_dataset_bytes` | Non-zero only at `charge_open_dataset_round` (default 1); equals `N_open × N_features × 4 × num_clients` |

The `total_wire` / `total_packed` `@property` helpers sum the three byte categories so the ledger can answer cumulative-bytes questions without re-aggregating.

**`CommCostLedger` class** — append-only log + cumulative query API:

| Method | Returns | Used by |
|---|---|---|
| `record(entry)` | `None` | Called once per round from `strategy.aggregate_fit` |
| `cumulative_bytes_at(round, packed=False)` | `int` | Internal — exact byte total up to & including `round` |
| `cumulative_mb_at(round, packed=False)` | `float` | Per-round `cumulative_mb_{wire,packed}` metric |
| `cumulative_series(packed=False)` | `List[(round, MB)]` | Input to the cumulative-MB-over-rounds plot in our report |
| `to_dict()` | `dict` | Serialised to `metrics/summary.json → comm_cost_ledger` |

**Key design choice — `×num_clients` fan-out on broadcast.** The paper counts broadcasts against every receiver; we match that convention so our `C@50` and `C@75` numbers are apples-to-apples with Zhao et al. Table IV. A future "centralized-view" variant could count only the single outbound broadcast; the ledger is trivially re-summable because the raw bytes are stored per category.

---

### 14.4 Summary extractors — Table III / Table IV cells (**live**)

All four extractors walk `strategy.round_metrics + strategy.comm_cost_ledger` — they never touch model state, so the tables can be regenerated from disk after a crashed run.

| Function | Returns | Table cell |
|---|---|---|
| `extract_top_acc(history)` | `(float, Optional[int])` — `(top_acc, round_of_top_acc)` | Table IV "Top-Acc" column |
| `extract_comm_cost_at_accuracy(history, ledger, target_acc, packed)` | `Optional[float]` MB at first round hitting `target_acc`; `None` if never reached | Table IV `C@50`, `C@75` |
| `extract_comm_cost_at_top_acc(history, ledger, packed)` | `Optional[float]` | Table IV `C@Top-Acc` |
| `extract_accuracy_snapshots(history, rounds)` | `Dict[int, Optional[float]]` | Table III row (`Top-1 @ {10,50,100,150,200}`) |
| `build_summary_report(history, ledger, target_accs, snapshot_rounds)` | aggregator `dict` | Combines all of the above into one serialisable payload |

**Accuracy key resolution order** (inside `_accuracy_of`): `server_eval_accuracy` → `accuracy` → `test_accuracy`. This lets pre-CNN runs emit `None` cleanly and post-CNN runs pick up the server-side evaluation automatically without the strategy having to be modified.

---

### 14.5 On-disk deliverables (**live**)

At `strategy` shutdown, `main.run_server` writes four files under `args.metrics_dir` (default `metrics/`) plus the Flower native history under `args.logs_dir`:

| File | Producer | Contents |
|---|---|---|
| `metrics/per_round.json` | `save_metrics_json(strategy.round_metrics, ...)` | Canonical indented-JSON dump of every round's full metric dict, including per-class lists and the round-by-round confusion matrix |
| `metrics/per_round.csv` | `save_metrics_csv(...)` | Scalar-only flattening (one row = one round). Ideal for quick `pd.read_csv` plotting |
| `metrics/summary.json` | `build_summary_report(...) + ledger.to_dict()` | Every Table II-IV cell — `top_acc`, `top_acc_round`, `c_at_top_acc_{wire,packed}_mb`, `c_at_open_dataset_{bytes,mb}`, `accuracy_snapshots`, `c_at_accuracy_{wire,packed}_mb`, `final_round_cumulative_{wire,packed}_mb`, and the full `comm_cost_ledger` blob |
| `metrics/confusion_matrix_final.json` | `save_confusion_matrix_json(...)` | Final-round confusion matrix + per-class F1/P/R/support + class names; backs Fig. 3 heatmap. Skipped (with a logged note) if `eval_fn` was stubbed for the whole run |
| `logs/history.json` | direct `json.dump` over Flower's `History` | `losses_distributed`, `metrics_distributed_fit`, `metrics_distributed`, plus a copy of `strategy.round_metrics` — kept because downstream tooling may depend on it |

---

### 14.6 Which metric maps to which paper cell (quick reference)

| Paper target | Our source | Notes |
|---|---|---|
| Table II accuracy | `summary.top_acc` | `eval_fn` must be live (post-CNN) |
| Table II F1 (macro) | `per_round.json[last].server_eval_f1_macro` | Derived from `compute_classification_metrics` |
| Table II precision / recall | `per_round.json[last].server_eval_{precision,recall}_macro` | Weighted variants also present |
| Table III Top-1 @ {10,50,100,150,200} | `summary.accuracy_snapshots` | Missing rounds → `None` |
| Table IV `C@Dᵒ` | `summary.c_at_open_dataset_mb` | Paper counts per-client; we follow the same convention |
| Table IV `C@50`, `C@75` | `summary.c_at_accuracy_packed_mb["0.50"/"0.75"]` | Use **packed** for paper-fair comparison; `wire` for real Flower cost |
| Table IV `C@Top-Acc` | `summary.c_at_top_acc_packed_mb` | Same packed-vs-wire choice applies |
| Table IV Top-Acc | `summary.top_acc` | Round where it first occurred → `summary.top_acc_round` |

---

### 14.7 Diagnostics we add beyond the paper

These do not appear in Zhao et al.'s tables but are kept because they materially improve our final-report story:

- `round_total_sec`, `server_vote_sec`, `avg_client_fit_sec`, `max_client_fit_sec`, `server_eval_sec` — wall-clock decomposition per round. Lets us defend the "SSFL is cheap in compute, not just bandwidth" angle.
- `avg_confidence_threshold` — the mean of clients' median θ values. Monitors drift of the familiar/unfamiliar split across rounds; a sudden upward or downward trend signals classifier over-confidence.
- `valid_global_labels` (per round) — how many of the `N_open` samples received any non-(-1) vote. Proxy for "how much of the open dataset is actually usable for distillation this round."
- Per-class F1 / precision / recall / support vectors — see §14.1 rationale.
- Wire vs. packed byte parity — auditable at every round because both are always logged.

---

*Plan authored for Capstone: Federated Pseudo Labeling for Semi-Supervised Intrusion Detection*
*Team: Ayberk Karataban, Demir Eroğlu, Kuzey Berk Yılmaz (CMP) | Cem Aksoy, Volkan Kısa, Doğa Özdür (SEN)*
*Framework: Flower (flwr) | Dataset: N-BaIoT | Scenario: Scenario 1 only*
