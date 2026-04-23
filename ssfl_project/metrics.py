"""Metrics & evaluation-protocol module — the single authority for every
number that appears in the final capstone report.

Design contract (mirrors `SSFL_FLOWER_INFRASTRUCTURE_PLAN.md` §14):

* Classification math (accuracy, precision, recall, F1, confusion matrix) is
  implemented *live* and has no CNN dependency — it operates on `y_true`
  and `y_pred` int arrays.
* Payload / byte-accounting helpers are *live* — they inspect numpy arrays
  and return byte counts. Two byte models are supported in parallel:
  `_WIRE` (actual Flower gRPC cost with int64/float32) and `_PAPER`
  (uint8-packed, matching Zhao et al. Table IV).
* The `CommCostLedger` is a *live* running total of uploaded / downloaded
  bytes per round; it exposes a `cumulative_mb_at(round)` lookup used by
  `extract_comm_cost_at_accuracy`.
* `C@x` / `Top-Acc` / `Top-1 @ round` extractors are *live* — they walk
  round-ordered history lists produced by the strategy.

Only `fl_baseline_upload_bytes` (which needs the real CNN's parameter
count) may raise `NotImplementedError` before model.py ships; until then
callers should pass `config.ESTIMATED_CNN_PARAM_COUNT` as a fallback.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Classification metrics — Table II analogue
# =====================================================================
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> dict:
    """Return every classification number the paper's Table II (and our report)
    needs, in a single dict.

    The dict is intentionally *flat* apart from per-class lists and the
    confusion matrix, so `save_metrics_csv` can dump it without further
    massaging. `y_true` and `y_pred` must be 1-D int arrays of identical
    length; entries outside `[0, num_classes)` are left unflagged (sklearn
    will just ignore them in the averaged stats), but they will appear in
    the confusion matrix row labelled "other" if `labels=` is widened.

    Returned keys:
        accuracy                -> float
        f1_macro, f1_weighted   -> float
        precision_macro, precision_weighted -> float
        recall_macro, recall_weighted       -> float
        f1_per_class            -> List[float]  length == num_classes
        precision_per_class     -> List[float]  length == num_classes
        recall_per_class        -> List[float]  length == num_classes
        support_per_class       -> List[int]    length == num_classes
        confusion_matrix        -> List[List[int]]  shape (num_classes, num_classes)
        class_names             -> List[str]    length == num_classes
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"compute_classification_metrics: y_true {y_true.shape} != "
            f"y_pred {y_pred.shape}"
        )

    labels_range: List[int] = list(range(num_classes))
    if class_names is None:
        class_names = [
            config.GLOBAL_ID_TO_CLASS_NAME.get(i, f"class_{i}")
            for i in labels_range
        ]

    acc: float = float(accuracy_score(y_true, y_pred))
    f1_macro: float = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    f1_weighted: float = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    prec_pc, rec_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, zero_division=0
    )
    prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, average="weighted", zero_division=0
    )
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=labels_range)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_per_class": [float(v) for v in f1_pc],
        "precision_per_class": [float(v) for v in prec_pc],
        "recall_per_class": [float(v) for v in rec_pc],
        "support_per_class": [int(v) for v in support_pc],
        "confusion_matrix": cm.astype(int).tolist(),
        "class_names": list(class_names),
    }


# =====================================================================
# 2. Payload sizing — Table IV analogue
# =====================================================================
def payload_bytes_wire(arr: np.ndarray) -> int:
    """Bytes that will actually leave the socket for this numpy array.

    Uses the array's dtype, so int64 hard-label arrays cost
    `N_open × 8` bytes even though the values fit in 4 bits — that's the
    real Flower/gRPC overhead when we don't manually quantize.
    """
    return int(arr.nbytes)


def payload_bytes_packed(arr: np.ndarray) -> int:
    """Paper-fair byte count assuming each label is packed into a uint8.

    Zhao et al. report SSFL upload cost on the order of 0.01 MB per round,
    which corresponds to `N_open × 1 byte`. This function matches that
    accounting so our Table IV analogue is directly comparable.
    """
    return int(arr.size) * config.BYTES_PER_HARD_LABEL_PAPER


def open_dataset_distribution_bytes(
    n_open: int,
    n_features: int = config.N_FEATURES,
    bytes_per_value: int = config.BYTES_PER_FLOAT32,
) -> int:
    """One-time `C@D^o` cost: server pushes the open dataset to every client.

    Counted once across the whole run (round 0 or "pre-training"). Because
    the open dataset is the same for every client, most implementations
    charge this as a single-payload amortization — but the paper counts
    it per-client, so we let callers multiply by `num_clients` if they
    want the distributed footprint.
    """
    return int(n_open) * int(n_features) * int(bytes_per_value)


def fl_baseline_upload_bytes(
    num_params: int = config.ESTIMATED_CNN_PARAM_COUNT,
    bytes_per_param: int = config.BYTES_PER_FLOAT32,
) -> int:
    """Per-client per-round upload cost of vanilla FL (parameter averaging).

    Used for the "FL" row in our Table IV analogue. When `num_params` is
    supplied from an actual CNN (`sum(p.numel() for p in classifier.parameters())`)
    this is exact; the default pulls the configured estimate so tables
    can still be produced before model.py ships.
    """
    return int(num_params) * int(bytes_per_param)


def bytes_to_mb(n_bytes: int) -> float:
    """Convert bytes to megabytes using the decimal MB definition (10^6)
    used in Zhao et al.'s Table IV."""
    return float(n_bytes) / 1_000_000.0


# =====================================================================
# 3. Running ledger — per-round + cumulative
# =====================================================================
@dataclass
class RoundCommCost:
    """Byte-level record for one communication round.

    `uploaded_bytes` sums hard-label payloads across all participating
    clients. `downloaded_bytes` is the global-label broadcast expanded
    by `num_clients` (the server sends to every client, so the total
    on-wire cost is `N_open × bytes_per_label × num_clients`).
    `open_dataset_bytes` is usually 0 except on round 1 (or round 0)
    where we charge the one-shot distribution.
    """
    round: int
    uploaded_bytes_wire: int
    uploaded_bytes_packed: int
    downloaded_bytes_wire: int
    downloaded_bytes_packed: int
    open_dataset_bytes: int = 0

    @property
    def total_wire(self) -> int:
        return (
            self.uploaded_bytes_wire
            + self.downloaded_bytes_wire
            + self.open_dataset_bytes
        )

    @property
    def total_packed(self) -> int:
        return (
            self.uploaded_bytes_packed
            + self.downloaded_bytes_packed
            + self.open_dataset_bytes
        )


class CommCostLedger:
    """Round-ordered accumulator. Query via `cumulative_mb_at(round)` or
    `cumulative_bytes_at(round)`; iterate via `entries`.
    """

    def __init__(self) -> None:
        self.entries: List[RoundCommCost] = []

    def record(self, entry: RoundCommCost) -> None:
        if self.entries and entry.round <= self.entries[-1].round:
            logger.warning(
                "CommCostLedger.record: non-monotonic round %d (previous %d)",
                entry.round,
                self.entries[-1].round,
            )
        self.entries.append(entry)

    # ---------- cumulative lookups ----------
    def cumulative_bytes_at(
        self, round_num: int, packed: bool = False
    ) -> int:
        total: int = 0
        for e in self.entries:
            if e.round > round_num:
                break
            total += e.total_packed if packed else e.total_wire
        return total

    def cumulative_mb_at(self, round_num: int, packed: bool = False) -> float:
        return bytes_to_mb(self.cumulative_bytes_at(round_num, packed=packed))

    def cumulative_series(
        self, packed: bool = False
    ) -> List[Tuple[int, float]]:
        """Return [(round, cumulative MB), ...] for plotting."""
        out: List[Tuple[int, float]] = []
        running: int = 0
        for e in self.entries:
            running += e.total_packed if packed else e.total_wire
            out.append((e.round, bytes_to_mb(running)))
        return out

    def to_dict(self) -> dict:
        return {
            "entries": [asdict(e) for e in self.entries],
            "cumulative_mb_wire": self.cumulative_mb_at(
                self.entries[-1].round if self.entries else 0, packed=False
            ),
            "cumulative_mb_packed": self.cumulative_mb_at(
                self.entries[-1].round if self.entries else 0, packed=True
            ),
        }


# =====================================================================
# 4. Summary extractors — populate the final Tables II-IV cells
# =====================================================================
def extract_top_acc(history: List[dict]) -> Tuple[float, Optional[int]]:
    """`(top_acc, round_number_of_top_acc)` over all rounds.

    Accepts dicts with either `server_eval_accuracy` (aggregated from
    eval_fn via the strategy) or `accuracy` (centralized eval path). Rows
    missing both are skipped. Returns `(0.0, None)` if the history has
    no usable accuracy rows.
    """
    best: float = -1.0
    best_round: Optional[int] = None
    for row in history:
        acc: Optional[float] = _accuracy_of(row)
        if acc is None:
            continue
        if acc > best:
            best = acc
            best_round = int(row.get("round", 0))
    if best_round is None:
        return 0.0, None
    return float(best), best_round


def extract_comm_cost_at_accuracy(
    history: List[dict],
    ledger: CommCostLedger,
    target_acc: float,
    packed: bool = False,
) -> Optional[float]:
    """Return cumulative MB at the **first** round whose test accuracy
    reaches or exceeds `target_acc`. Returns `None` if that accuracy is
    never reached.
    """
    for row in sorted(history, key=lambda r: int(r.get("round", 0))):
        acc = _accuracy_of(row)
        if acc is None:
            continue
        if acc >= target_acc:
            return ledger.cumulative_mb_at(int(row["round"]), packed=packed)
    return None


def extract_comm_cost_at_top_acc(
    history: List[dict], ledger: CommCostLedger, packed: bool = False
) -> Optional[float]:
    """Cumulative MB at the round where `top_acc` is first reached."""
    _, top_round = extract_top_acc(history)
    if top_round is None:
        return None
    return ledger.cumulative_mb_at(top_round, packed=packed)


def extract_accuracy_snapshots(
    history: List[dict],
    rounds: Tuple[int, ...] = config.SNAPSHOT_ROUNDS,
) -> Dict[int, Optional[float]]:
    """`{round: acc or None}` — populates the Table III "Top-1 @ round" row.

    For a requested round that wasn't actually run (or that had no eval),
    returns `None` rather than falsely interpolating.
    """
    by_round: Dict[int, float] = {}
    for row in history:
        acc = _accuracy_of(row)
        if acc is None:
            continue
        by_round[int(row.get("round", 0))] = acc
    return {r: by_round.get(int(r)) for r in rounds}


def build_summary_report(
    history: List[dict],
    ledger: CommCostLedger,
    target_accs: Tuple[float, ...] = config.TARGET_ACCURACIES,
    snapshot_rounds: Tuple[int, ...] = config.SNAPSHOT_ROUNDS,
) -> dict:
    """Assemble every cell that appears in our final Tables II/III/IV.

    Output schema (all numbers json-serialisable):

        top_acc                         -> float
        top_acc_round                   -> int | None
        c_at_top_acc_wire_mb            -> float | None
        c_at_top_acc_packed_mb          -> float | None
        c_at_open_dataset_bytes         -> int     (== sum of one-shot distribution costs)
        accuracy_snapshots              -> {round_int: acc or None}
        c_at_accuracy_wire_mb           -> {"0.50": float|None, "0.75": float|None, ...}
        c_at_accuracy_packed_mb         -> {...}
        final_round_cumulative_wire_mb  -> float
        final_round_cumulative_packed_mb-> float
    """
    top_acc, top_round = extract_top_acc(history)
    c_at_open_dataset: int = sum(e.open_dataset_bytes for e in ledger.entries)
    final_round: int = ledger.entries[-1].round if ledger.entries else 0

    c_at_acc_wire: Dict[str, Optional[float]] = {
        f"{t:.2f}": extract_comm_cost_at_accuracy(history, ledger, t, packed=False)
        for t in target_accs
    }
    c_at_acc_packed: Dict[str, Optional[float]] = {
        f"{t:.2f}": extract_comm_cost_at_accuracy(history, ledger, t, packed=True)
        for t in target_accs
    }

    return {
        "top_acc": top_acc,
        "top_acc_round": top_round,
        "c_at_top_acc_wire_mb": extract_comm_cost_at_top_acc(
            history, ledger, packed=False
        ),
        "c_at_top_acc_packed_mb": extract_comm_cost_at_top_acc(
            history, ledger, packed=True
        ),
        "c_at_open_dataset_bytes": int(c_at_open_dataset),
        "c_at_open_dataset_mb": bytes_to_mb(c_at_open_dataset),
        "accuracy_snapshots": {
            int(k): (float(v) if v is not None else None)
            for k, v in extract_accuracy_snapshots(history, snapshot_rounds).items()
        },
        "c_at_accuracy_wire_mb": c_at_acc_wire,
        "c_at_accuracy_packed_mb": c_at_acc_packed,
        "final_round_cumulative_wire_mb": ledger.cumulative_mb_at(
            final_round, packed=False
        ),
        "final_round_cumulative_packed_mb": ledger.cumulative_mb_at(
            final_round, packed=True
        ),
    }


# =====================================================================
# 5. I/O helpers — write to disk for later plotting / report-writing
# =====================================================================
def save_metrics_json(history: List[dict], output_path: str) -> None:
    """Dump full per-round history as indented JSON (canonical form)."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2, default=_json_fallback)


def save_metrics_csv(history: List[dict], output_path: str) -> None:
    """Flatten per-round dicts (drop list-valued per-class fields and the
    confusion matrix; those live in the JSON). Produces a tidy CSV where
    one row == one round, one column == one scalar metric.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not history:
        with open(output_path, "w", newline="") as f:
            f.write("")
        return

    # Collect scalar keys across all rows (schema may expand round-to-round).
    scalar_keys: List[str] = []
    for row in history:
        for k, v in row.items():
            if k in scalar_keys:
                continue
            if isinstance(v, (int, float, bool)) or v is None:
                scalar_keys.append(k)

    # Keep `round` first if present, then alphabetical.
    if "round" in scalar_keys:
        scalar_keys.remove("round")
    scalar_keys = ["round"] + sorted(scalar_keys) if "round" in {k for r in history for k in r} else sorted(scalar_keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in scalar_keys})


def save_summary_json(summary: dict, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_fallback)


def save_confusion_matrix_json(
    classification_metrics: dict, output_path: str
) -> None:
    """Persist just the confusion matrix + class names — this is the input
    to the Fig. 3 heatmap we'll generate in the final report.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "confusion_matrix": classification_metrics.get("confusion_matrix"),
        "class_names": classification_metrics.get("class_names"),
        "f1_per_class": classification_metrics.get("f1_per_class"),
        "precision_per_class": classification_metrics.get("precision_per_class"),
        "recall_per_class": classification_metrics.get("recall_per_class"),
        "support_per_class": classification_metrics.get("support_per_class"),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_fallback)


# =====================================================================
# 6. Internal helpers
# =====================================================================
def _accuracy_of(row: dict) -> Optional[float]:
    """Extract accuracy from a round-metric row, trying known key names in
    priority order. Returns None if no accuracy is recorded for the row."""
    for key in (
        "server_eval_accuracy",   # from strategy.aggregate_fit via eval_fn
        "accuracy",               # centralized eval
        "test_accuracy",          # legacy / custom
    ):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def _json_fallback(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
