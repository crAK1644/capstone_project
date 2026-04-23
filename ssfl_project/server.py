"""Flower server startup + server-side evaluation closure.

`start_server` is fully implemented. `build_eval_fn` returns a closure
whose signature is final; the closure body raises NotImplementedError
until the CNN in model.py is delivered.
"""
import logging
from typing import Callable

import flwr as fl
import numpy as np
import torch

from strategy import SSFLStrategy

logger = logging.getLogger(__name__)


# ---------- 8.1 ----------
def build_eval_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> Callable:
    """Build a closure that evaluates a server-side global model on the test set.

    The closure has signature:
        eval_fn(server_round: int, global_labels: np.ndarray, X_open) -> dict

    Pass 2 implementation will, inside the closure:
        - build a fresh TrafficCNN classifier
        - train it on (X_open[valid], global_labels[valid]) for a few epochs
        - evaluate on (X_test, y_test)
        - return {'accuracy': ..., 'f1_macro': ..., 'precision_macro': ...,
                  'recall_macro': ...}

    The SSFLStrategy catches NotImplementedError so the server loop still
    runs when the closure body is stubbed.
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
