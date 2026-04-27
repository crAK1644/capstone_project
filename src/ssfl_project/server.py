"""Flower server startup + server-side evaluation closure.

`start_server` is fully implemented. `build_eval_fn` returns a closure
whose signature is final; the closure body raises NotImplementedError
until the CNN in model.py is delivered.

The eval_fn's returned dict is prefixed with `server_eval_` inside
`SSFLStrategy.aggregate_fit` and written into `strategy.round_metrics`;
downstream, `metrics.build_summary_report` walks that history to fill
our Table II / III / IV analogues. See `SSFL_FLOWER_INFRASTRUCTURE_PLAN.md`
§14 for the end-to-end metrics contract.
"""
import logging
from typing import Callable

import flwr as fl
import numpy as np
import torch

import config
from metrics import compute_classification_metrics  # noqa: F401  (used by the live body post-CNN)
from strategy import SSFLStrategy

logger = logging.getLogger(__name__)


# ---------- 8.1 ----------
def build_eval_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    server_eval_epochs: int = config.SERVER_EVAL_EPOCHS,
) -> Callable:
    """Build a closure that evaluates a server-side global model on the test set.

    The closure has signature:
        eval_fn(server_round: int, global_labels: np.ndarray, X_open) -> dict

    Pass 2 implementation (after model.py ships) will, inside the closure:
        1. Mask `global_labels` to only the rows with valid (!= -1) votes.
        2. Build a fresh `TrafficCNN` classifier.
        3. Train it on `(X_open[valid], global_labels[valid])` for
           `server_eval_epochs` epochs (default from config.SERVER_EVAL_EPOCHS).
        4. Run it on `(X_test, y_test)` to produce a y_pred array.
        5. Call `metrics.compute_classification_metrics(y_test, y_pred,
           num_classes)` and return its dict verbatim.

    Expected return schema (every key becomes a `server_eval_<key>` field
    in the strategy's `round_metrics`, mirroring `metrics.compute_classification_metrics`):

        accuracy                -> float
        f1_macro                -> float
        f1_weighted             -> float
        precision_macro         -> float
        precision_weighted      -> float
        recall_macro            -> float
        recall_weighted         -> float
        f1_per_class            -> List[float]           (length num_classes)
        precision_per_class     -> List[float]
        recall_per_class        -> List[float]
        support_per_class       -> List[int]
        confusion_matrix        -> List[List[int]]       (num_classes × num_classes)
        class_names             -> List[str]

    The strategy catches `NotImplementedError` so the server loop still
    advances (with missing accuracy fields on that round) when the closure
    body is stubbed. That keeps the pre-CNN run useful for verifying
    communication accounting and voting correctness end-to-end.
    """

    def eval_fn(
        server_round: int,
        global_labels: np.ndarray,
        X_open,  # shape (N_open, 23, 5) when provided; may be None in pass-1 wiring
    ) -> dict:
        raise NotImplementedError(
            "build_eval_fn closure body deferred until TrafficCNN is implemented."
        )

    return eval_fn


# ---------- 8.2 ----------
def start_server(
    server_address: str,
    strategy: SSFLStrategy,
    num_rounds: int,
) -> fl.server.History:
    """Launch the Flower server for `num_rounds` communication rounds."""
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    logger.info(
        "Starting Flower server on %s for %d rounds (num_clients=%d)",
        server_address,
        num_rounds,
        strategy.num_clients,
    )
    history: fl.server.History = fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=server_config,
    )
    return history
