"""N-BaIoT dataset loading, normalization, and Scenario 1 partitioning.

Fully implemented — no CNN dependency. Run once before training:

    python data_preparation.py --raw_dir data/raw --output_dir data/partitions
"""
import glob
import logging
import os
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from utils import get_feature_column_names

logger = logging.getLogger(__name__)


# ---------- CSV filename -> class name ----------
def _extract_class_name(csv_path: str) -> str:
    """Extract canonical class name (e.g., 'mirai_udp') from a CSV filename.

    Handles minor variations: dots/hyphens become underscores, trailing
    integer suffixes (e.g., `_1`) are stripped, case-insensitive.
    """
    stem: str = os.path.splitext(os.path.basename(csv_path))[0].lower()
    stem = stem.replace(".", "_").replace("-", "_").strip("_")
    stem = re.sub(r"_\d+$", "", stem)
    return stem


# ---------- 4.1 ----------
def load_device_csvs(device_dir: str) -> pd.DataFrame:
    """Load all CSVs in one device folder; attach global integer labels."""
    all_frames: List[pd.DataFrame] = []
    csv_files: List[str] = sorted(glob.glob(os.path.join(device_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {device_dir!r}")

    for csv_path in csv_files:
        class_name: str = _extract_class_name(csv_path)
        if class_name not in config.CLASS_NAME_TO_GLOBAL_ID:
            raise KeyError(
                f"Unknown class name {class_name!r} from {csv_path!r}. "
                f"Known keys: {sorted(config.CLASS_NAME_TO_GLOBAL_ID)}"
            )
        global_label: int = config.CLASS_NAME_TO_GLOBAL_ID[class_name]
        df: pd.DataFrame = pd.read_csv(csv_path)
        df["label"] = global_label
        all_frames.append(df)

    device_df: pd.DataFrame = pd.concat(all_frames, axis=0).reset_index(drop=True)
    return device_df


# ---------- 4.2 ----------
def build_mini_nbaiot(
    raw_data_dir: str,
    samples_per_class: int = config.SAMPLES_PER_CLASS,
) -> pd.DataFrame:
    """Build the mini dataset: `samples_per_class` rows per (device, class)."""
    device_dirs: List[str] = sorted(
        d for d in glob.glob(os.path.join(raw_data_dir, "*")) if os.path.isdir(d)
    )
    if len(device_dirs) == 0:
        raise FileNotFoundError(f"No device directories in {raw_data_dir!r}")

    sampled_frames: List[pd.DataFrame] = []
    for device_id, device_dir in enumerate(device_dirs):
        device_df: pd.DataFrame = load_device_csvs(device_dir)
        device_df["device_id"] = device_id
        class_labels: np.ndarray = device_df["label"].unique()
        for label in class_labels:
            class_subset: pd.DataFrame = device_df[device_df["label"] == label]
            n: int = len(class_subset)
            replace: bool = n < samples_per_class
            sampled_class: pd.DataFrame = class_subset.sample(
                n=samples_per_class,
                replace=replace,
                random_state=config.RANDOM_SEED,
            )
            sampled_frames.append(sampled_class)

    mini_df: pd.DataFrame = pd.concat(sampled_frames, axis=0).reset_index(drop=True)
    return mini_df


# ---------- 4.3 ----------
def normalize_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, dict]:
    """Min-max normalize `feature_cols` to [0, 1]. Returns (df_norm, norm_params)."""
    df_copy: pd.DataFrame = df.copy()
    feature_min: pd.Series = df_copy[feature_cols].min()
    feature_max: pd.Series = df_copy[feature_cols].max()
    range_vals: pd.Series = (feature_max - feature_min).replace(0, 1.0)
    df_copy[feature_cols] = (df_copy[feature_cols] - feature_min) / range_vals
    norm_params: dict = {"min": feature_min, "max": feature_max}
    return df_copy, norm_params


def apply_normalization(
    df: pd.DataFrame, feature_cols: List[str], norm_params: dict
) -> pd.DataFrame:
    """Apply previously-computed min/max to a new DataFrame (e.g., test split)."""
    df_copy: pd.DataFrame = df.copy()
    feature_min: pd.Series = norm_params["min"]
    feature_max: pd.Series = norm_params["max"]
    range_vals: pd.Series = (feature_max - feature_min).replace(0, 1.0)
    df_copy[feature_cols] = (df_copy[feature_cols] - feature_min) / range_vals
    return df_copy


# ---------- 4.4 ----------
def reshape_sample_to_2d(sample_vector: np.ndarray) -> np.ndarray:
    """Reshape flat (115,) vector -> (23, 5) matrix.

    Column j of the output holds features from time window j; row k holds the
    k-th within-window feature. Flat index j*23 + k maps to matrix[k, j].
    """
    matrix: np.ndarray = np.zeros(
        (config.FEATURES_PER_WINDOW, config.N_TIME_WINDOWS), dtype=np.float32
    )
    for time_window_idx in range(config.N_TIME_WINDOWS):
        feature_offset: int = time_window_idx * config.FEATURES_PER_WINDOW
        matrix[:, time_window_idx] = sample_vector[
            feature_offset : feature_offset + config.FEATURES_PER_WINDOW
        ]
    return matrix


# ---------- 4.5 ----------
def apply_2d_reshape_to_dataset(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply reshape to every row. Returns X of shape (N, 23, 5) and y of (N,)."""
    feature_matrix: np.ndarray = df[feature_cols].to_numpy(dtype=np.float32)
    N: int = feature_matrix.shape[0]
    X: np.ndarray = np.zeros(
        (N, config.FEATURES_PER_WINDOW, config.N_TIME_WINDOWS), dtype=np.float32
    )
    for i in range(N):
        X[i] = reshape_sample_to_2d(feature_matrix[i])
    if "label" in df.columns:
        y: np.ndarray = df["label"].to_numpy(dtype=np.int64)
    else:
        y = np.full((N,), -1, dtype=np.int64)
    return X, y


# ---------- 4.6 ----------
def split_private_open_test(
    mini_df: pd.DataFrame,
    private_ratio: float = config.PRIVATE_RATIO,
    open_ratio: float = config.OPEN_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by (device_id, label). Open split drops its label column."""
    if abs(private_ratio + open_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("private + open + test ratios must sum to 1.0")

    rng: np.random.Generator = np.random.default_rng(random_seed)
    private_parts: List[pd.DataFrame] = []
    open_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for (_dev_id, _label), group in mini_df.groupby(["device_id", "label"]):
        n: int = len(group)
        shuffled_idx: np.ndarray = rng.permutation(n)
        shuffled_group: pd.DataFrame = group.iloc[shuffled_idx].reset_index(drop=True)
        n_private: int = int(n * private_ratio)
        n_open: int = int(n * open_ratio)
        # n_test computed as remainder to avoid float rounding drift
        private_parts.append(shuffled_group.iloc[:n_private])
        open_parts.append(shuffled_group.iloc[n_private : n_private + n_open])
        test_parts.append(shuffled_group.iloc[n_private + n_open :])

    private_df: pd.DataFrame = pd.concat(private_parts, axis=0).reset_index(drop=True)
    open_df: pd.DataFrame = (
        pd.concat(open_parts, axis=0).reset_index(drop=True).drop(columns=["label"])
    )
    test_df: pd.DataFrame = pd.concat(test_parts, axis=0).reset_index(drop=True)
    return private_df, open_df, test_df


# ---------- 4.7 ----------
def partition_scenario1(
    private_df: pd.DataFrame,
    target_device_id: int,
    k_di: int = config.K_DI,
) -> List[pd.DataFrame]:
    """Shard-based non-IID partitioning for one device.

    Sorts the device's private rows by label, splits into `2 * k_di` equal
    shards, and assigns 2 consecutive shards to each of `k_di` clients.
    """
    device_data: pd.DataFrame = (
        private_df[private_df["device_id"] == target_device_id].reset_index(drop=True)
    )
    sorted_data: pd.DataFrame = (
        device_data.sort_values("label", kind="stable").reset_index(drop=True)
    )
    n_total: int = len(sorted_data)
    n_shards: int = 2 * k_di
    shard_size: int = n_total // n_shards  # remainder samples discarded for even shards
    if shard_size == 0:
        raise ValueError(
            f"Device {target_device_id}: only {n_total} rows; cannot make "
            f"{n_shards} non-empty shards."
        )
    shards: List[pd.DataFrame] = [
        sorted_data.iloc[i * shard_size : (i + 1) * shard_size].reset_index(drop=True)
        for i in range(n_shards)
    ]
    client_datasets: List[pd.DataFrame] = []
    for i in range(k_di):
        client_df: pd.DataFrame = pd.concat(
            [shards[2 * i], shards[2 * i + 1]], axis=0
        ).reset_index(drop=True)
        client_datasets.append(client_df)
    return client_datasets


# ---------- 4.8 ----------
def build_all_client_partitions(
    private_df: pd.DataFrame,
    k_di: int = config.K_DI,
) -> Dict[int, pd.DataFrame]:
    """Build all 27 client partitions keyed by global client id (0..26)."""
    all_partitions: Dict[int, pd.DataFrame] = {}
    device_ids: List[int] = sorted(int(d) for d in private_df["device_id"].unique())
    for device_id in device_ids:
        device_partitions: List[pd.DataFrame] = partition_scenario1(
            private_df, target_device_id=device_id, k_di=k_di
        )
        for local_client_id, client_df in enumerate(device_partitions):
            global_client_id: int = device_id * k_di + local_client_id
            all_partitions[global_client_id] = client_df
    return all_partitions


# ---------- 4.9 ----------
def save_partitions(
    all_partitions: Dict[int, pd.DataFrame],
    open_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Serialize partitions + open + test to pickle files under `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    for client_id, df in all_partitions.items():
        file_path: str = os.path.join(output_dir, f"client_{client_id}_private.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(df, f)
    open_path: str = os.path.join(output_dir, "open_data.pkl")
    with open(open_path, "wb") as f:
        pickle.dump(open_df, f)
    test_path: str = os.path.join(output_dir, "test_data.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(test_df, f)


# ---------- 4.10 ----------
def load_client_partition(
    client_id: int, partition_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one client's private partition as (X_private, y_private) arrays."""
    file_path = os.path.join(partition_dir, f"client_{client_id}_private.pkl")
    
    # --- GÜVENLİ BYPASS VE BOYUT SIKIŞTIRMA ---
    if not os.path.exists(file_path):
        client_dir = os.path.join("prepared_data", "scenario_1", f"client_{client_id}")
        opt_x = os.path.join(client_dir, "X_2d.npy")
        opt_y = os.path.join(client_dir, "y.npy")
        if os.path.exists(opt_x):
            X = np.load(opt_x)
            if X.ndim == 4:
                X = np.squeeze(X, axis=1) # 4D -> 3D dönüşümü (100, 1, 23, 5 -> 100, 23, 5)
            return X, np.load(opt_y)
    # -----------------------------------------
            
    with open(file_path, "rb") as f:
        client_df: pd.DataFrame = pickle.load(f)
    feature_cols: List[str] = get_feature_column_names()
    return apply_2d_reshape_to_dataset(client_df, feature_cols)


# ---------- 4.11 ----------
def load_open_data(partition_dir: str) -> np.ndarray:
    """Load the shared unlabeled open dataset as X_open of shape (N_open, 23, 5)."""
    open_path = os.path.join(partition_dir, "open_data.pkl")
    
    # --- GÜVENLİ BYPASS VE BOYUT SIKIŞTIRMA ---
    if not os.path.exists(open_path):
        opt_x = os.path.join("prepared_data", "open", "X_2d.npy")
        if os.path.exists(opt_x):
            X = np.load(opt_x)
            if X.ndim == 4:
                X = np.squeeze(X, axis=1)
            print("[Mimari] Ortak veri (Open) 3D formata sıkıştırılarak yüklendi.")
            return X
    # -----------------------------------------
            
    with open(open_path, "rb") as f:
        open_df: pd.DataFrame = pickle.load(f)
    feature_cols: List[str] = get_feature_column_names()
    X_open, _ = apply_2d_reshape_to_dataset(open_df, feature_cols)
    return X_open


def load_test_data(partition_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the test dataset as (X_test, y_test) arrays."""
    test_path = os.path.join(partition_dir, "test_data.pkl")
    
    # --- GÜVENLİ BYPASS VE BOYUT SIKIŞTIRMA ---
    if not os.path.exists(test_path):  
        opt_x = os.path.join("prepared_data", "test", "X_2d.npy")
        opt_y = os.path.join("prepared_data", "test", "y.npy")
        if os.path.exists(opt_x):
            X = np.load(opt_x)
            if X.ndim == 4:
                X = np.squeeze(X, axis=1)
            print("[Mimari] Test verisi 3D formata sıkıştırılarak yüklendi.")
            return X, np.load(opt_y)
    # -----------------------------------------
            
    with open(test_path, "rb") as f:
        test_df: pd.DataFrame = pickle.load(f)
    feature_cols: List[str] = get_feature_column_names()
    return apply_2d_reshape_to_dataset(test_df, feature_cols)


# ---------- 4.12 ----------
def create_torch_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader."""
    X_tensor: torch.Tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor: torch.Tensor = torch.as_tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------- one-shot partition generation ----------
def main() -> None:
    """End-to-end: raw CSVs -> mini -> normalize -> split -> shard -> pickle."""
    import argparse

    from utils import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(
        description="Prepare SSFL partitions (one-shot data pipeline)"
    )
    parser.add_argument("--raw_dir", default=config.RAW_DIR)
    parser.add_argument("--output_dir", default=config.PARTITION_DIR)
    parser.add_argument(
        "--samples_per_class", type=int, default=config.SAMPLES_PER_CLASS
    )
    args = parser.parse_args()

    logger.info("Building mini-N-BaIoT from %s", args.raw_dir)
    mini_df: pd.DataFrame = build_mini_nbaiot(
        args.raw_dir, samples_per_class=args.samples_per_class
    )
    logger.info("mini-N-BaIoT: %d rows, %d cols", len(mini_df), len(mini_df.columns))

    canonical_cols: List[str] = get_feature_column_names()
    present_cols: List[str] = [c for c in canonical_cols if c in mini_df.columns]
    if len(present_cols) != len(canonical_cols):
        logger.warning(
            "Only %d of %d canonical feature columns found in CSVs. "
            "Using present subset; you may need to adjust naming in config.py.",
            len(present_cols),
            len(canonical_cols),
        )

    mini_df_norm, _norm_params = normalize_features(mini_df, present_cols)
    logger.info("Normalized %d features", len(present_cols))

    private_df, open_df, test_df = split_private_open_test(mini_df_norm)
    logger.info(
        "Splits: private=%d, open=%d, test=%d",
        len(private_df), len(open_df), len(test_df),
    )

    partitions: Dict[int, pd.DataFrame] = build_all_client_partitions(private_df)
    logger.info("Built %d client partitions", len(partitions))

    save_partitions(partitions, open_df, test_df, args.output_dir)
    logger.info("Saved partitions to %s", args.output_dir)


if __name__ == "__main__":
    main()
