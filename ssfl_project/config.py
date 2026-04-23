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
METRICS_DIR: str = "metrics"
LOGS_DIR: str = "logs"

# ---------- Evaluation protocol (see SSFL_FLOWER_INFRASTRUCTURE_PLAN.md §14) ----------
# Paper Table III reports Top-1 accuracy at these rounds; we snapshot at the
# same checkpoints so our results table mirrors Zhao et al. one-for-one.
SNAPSHOT_ROUNDS: Tuple[int, ...] = (10, 50, 100, 150, 200)

# Paper Table IV reports cumulative MB to reach these accuracies (C@x); the
# metrics module walks round history to find the first crossing.
TARGET_ACCURACIES: Tuple[float, ...] = (0.50, 0.75)

# ---------- Byte-accounting for communication cost (§14.3) ----------
# `_WIRE` values reflect what actually leaves the socket in our Flower
# implementation; `_PAPER` values reflect the minimum-bit encoding Zhao et al.
# use in their communication-overhead comparison (Table IV). We log both so
# the team can pick whichever the final report needs.
BYTES_PER_FLOAT32: int = 4
BYTES_PER_INT64: int = 8
BYTES_PER_HARD_LABEL_WIRE: int = 8   # numpy default int64 over gRPC
BYTES_PER_HARD_LABEL_PAPER: int = 1  # uint8 packed (4 bits would suffice for 12 values)
BYTES_PER_OPEN_SAMPLE_FP32: int = N_FEATURES * BYTES_PER_FLOAT32  # 115 * 4 = 460
BYTES_PER_OPEN_SAMPLE_UINT8: int = N_FEATURES                     # paper-fair quantized baseline

# Estimate for the FL baseline (parameter-averaging) upload size per client.
# Computed from the 8-layer CNN in §5.1. Used by metrics.py when the CNN is
# still stubbed so we can still print "what FL *would* have cost" in reports.
# The live value should be recomputed from `sum(p.numel() for p in classifier.parameters())`
# once model.py ships.
ESTIMATED_CNN_PARAM_COUNT: int = 300_000  # conservative rounded estimate for 8-conv + 2-FC net

# ---------- Evaluation model settings ----------
# Epochs the server-side eval_fn uses when training a fresh CNN on
# (X_open[valid], global_labels[valid]) before scoring on X_test.
SERVER_EVAL_EPOCHS: int = 10

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
