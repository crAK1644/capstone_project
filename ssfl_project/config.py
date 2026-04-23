"""Single source of truth for all SSFL hyperparameters, paths, and dataset constants.

Every other module imports from here instead of hard-coding numbers or strings.
`main.parse_arguments()` wires its --flag defaults to these constants so CLI and
code cannot drift apart.
"""
import os
from typing import Dict, List, Tuple

# ---------- Core topology ----------
NUM_DEVICES: int = 9
K_DI: int = 3
NUM_CLIENTS: int = NUM_DEVICES * K_DI  # 27 in Scenario 1
NUM_CLASSES: int = 11  # global label space: 1 benign + 10 attack families

# ---------- Feature dimensions ----------
N_FEATURES: int = 115
N_TIME_WINDOWS: int = 5
FEATURES_PER_WINDOW: int = 23
INPUT_SHAPE: Tuple[int, int] = (FEATURES_PER_WINDOW, N_TIME_WINDOWS)  # (23, 5)

# ---------- Mini-N-BaIoT ----------
SAMPLES_PER_CLASS: int = 1000

# ---------- Splits (must sum to 1.0) ----------
PRIVATE_RATIO: float = 0.70
OPEN_RATIO: float = 0.10
TEST_RATIO: float = 0.20

# ---------- Training hyperparameters ----------
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 100
CLASSIFIER_EPOCHS: int = 5
DISCRIMINATOR_EPOCHS: int = 5
DISTILLATION_EPOCHS: int = 5
NUM_ROUNDS: int = 150
RANDOM_SEED: int = 42

# ---------- Networking ----------
DEFAULT_SERVER_ADDRESS: str = "127.0.0.1:8080"

# ---------- Filesystem ----------
DATA_DIR: str = "data"
RAW_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
PARTITION_DIR: str = os.path.join(DATA_DIR, "partitions")

# ---------- Class name -> global ID lookup ----------
# Authoritative mapping from traffic-category filename stem to global integer
# label in 0..10. `data_preparation.load_device_csvs` consults this to guarantee
# the same attack family gets the same integer across all 9 devices, even when
# a device is missing some classes.
CLASS_NAME_TO_GLOBAL_ID: Dict[str, int] = {
    "benign":          0,
    "gafgyt_combo":    1,
    "gafgyt_junk":     2,
    "gafgyt_scan":     3,
    "gafgyt_tcp":      4,
    "gafgyt_udp":      5,
    "mirai_ack":       6,
    "mirai_scan":      7,
    "mirai_syn":       8,
    "mirai_udp":       9,
    "mirai_udpplain": 10,
}
GLOBAL_ID_TO_CLASS_NAME: Dict[int, str] = {
    v: k for k, v in CLASS_NAME_TO_GLOBAL_ID.items()
}


# ---------- Canonical N-BaIoT feature column names ----------
# Ordering: grouped by time-decay level first, then by category, then by
# statistic. Flat index `j*23 + k` maps to (feature k, time window j) in the
# reshape produced by `data_preparation.reshape_sample_to_2d`.
_TIME_LEVELS: Tuple[str, ...] = ("L5", "L3", "L1", "L0.1", "L0.01")


def _build_feature_column_names() -> List[str]:
    names: List[str] = []
    for L in _TIME_LEVELS:
        # MI_dir: 3 stats
        names.extend([
            f"MI_dir_{L}_weight", f"MI_dir_{L}_mean", f"MI_dir_{L}_variance",
        ])
        # H: 3 stats
        names.extend([
            f"H_{L}_weight", f"H_{L}_mean", f"H_{L}_variance",
        ])
        # HH: 7 stats
        names.extend([
            f"HH_{L}_weight", f"HH_{L}_mean", f"HH_{L}_std",
            f"HH_{L}_magnitude", f"HH_{L}_radius",
            f"HH_{L}_covariance", f"HH_{L}_pcc",
        ])
        # HH_jit: 3 stats
        names.extend([
            f"HH_jit_{L}_weight", f"HH_jit_{L}_mean", f"HH_jit_{L}_variance",
        ])
        # HpHp: 7 stats
        names.extend([
            f"HpHp_{L}_weight", f"HpHp_{L}_mean", f"HpHp_{L}_std",
            f"HpHp_{L}_magnitude", f"HpHp_{L}_radius",
            f"HpHp_{L}_covariance", f"HpHp_{L}_pcc",
        ])
    return names


FEATURE_COLUMN_NAMES: List[str] = _build_feature_column_names()
assert len(FEATURE_COLUMN_NAMES) == N_FEATURES, (
    f"config.FEATURE_COLUMN_NAMES length mismatch: "
    f"expected {N_FEATURES}, got {len(FEATURE_COLUMN_NAMES)}"
)
assert abs(PRIVATE_RATIO + OPEN_RATIO + TEST_RATIO - 1.0) < 1e-9, (
    "PRIVATE_RATIO + OPEN_RATIO + TEST_RATIO must equal 1.0"
)
