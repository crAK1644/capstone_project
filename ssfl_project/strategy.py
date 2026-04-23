"""Custom Flower Strategy: voting-based aggregation over hard labels.

Fully implemented — no CNN dependency. The voting mechanism is the core
server-side algorithm of SSFL and this module can be unit-tested in
isolation.
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import numpy as np

logger = logging.getLogger(__name__)


class SSFLStrategy(Strategy):
    """Server-side strategy: collect hard-label arrays from all clients,
    run majority voting to produce a global label array, broadcast it."""

    # ---------- 7.1 Constructor ----------
    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        n_open_samples: int,
        min_fit_clients: int,
        min_available_clients: int,
        eval_fn: Optional[Callable] = None,
        classifier_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.num_clients: int = num_clients
        self.num_classes: int = num_classes
        self.n_open_samples: int = n_open_samples
        self.min_fit_clients: int = min_fit_clients
        self.min_available_clients: int = min_available_clients
        self.eval_fn: Optional[Callable] = eval_fn
        self.classifier_epochs: int = classifier_epochs

        # Initial global labels: all -1 (no consensus yet).
        self.global_labels: np.ndarray = np.full(
            (n_open_samples,), -1, dtype=np.int64
        )
        self.round_metrics: List[dict] = []

    # ---------- 7.2 ----------
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_labels: np.ndarray = self.global_labels.copy()
        return ndarrays_to_parameters([initial_labels])

    # ---------- 7.3 ----------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {
            "round": int(server_round),
            "classifier_epochs": int(self.classifier_epochs),
        }
        global_labels_params: Parameters = ndarrays_to_parameters(
            [self.global_labels.copy()]
        )
        fit_ins = FitIns(global_labels_params, config)
        clients: List[ClientProxy] = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    # ---------- 7.4 ----------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            logger.warning(
                "[Round %d] %d client failure(s)", server_round, len(failures)
            )
        if not results:
            logger.error("[Round %d] no fit results; holding prior global labels", server_round)
            return ndarrays_to_parameters([self.global_labels.copy()]), {}

        all_client_labels: List[np.ndarray] = []
        client_metrics: List[dict] = []
        for _client_proxy, fit_res in results:
            arrs: List[np.ndarray] = parameters_to_ndarrays(fit_res.parameters)
            if not arrs:
                logger.warning(
                    "[Round %d] client returned empty parameter list; skipping",
                    server_round,
                )
                continue
            all_client_labels.append(arrs[0].astype(np.int64))
            client_metrics.append(dict(fit_res.metrics))

        new_global_labels: np.ndarray = self.vote_mechanism(all_client_labels)
        self.global_labels = new_global_labels

        # Aggregate client metrics
        agg_metrics: Dict[str, Scalar] = {}
        for key in ("classifier_loss", "discriminator_loss", "distillation_loss"):
            vals: List[float] = [
                float(m.get(key, 0.0)) for m in client_metrics if key in m
            ]
            agg_metrics[f"avg_{key}"] = float(np.mean(vals)) if vals else 0.0
        agg_metrics["total_familiar"] = int(
            sum(int(m.get("n_familiar", 0)) for m in client_metrics)
        )
        agg_metrics["total_unfamiliar"] = int(
            sum(int(m.get("n_unfamiliar", 0)) for m in client_metrics)
        )
        agg_metrics["valid_global_labels"] = int((new_global_labels != -1).sum())
        agg_metrics["round"] = int(server_round)

        # Optional server-side evaluation (closure body is stubbed until CNN lands).
        if self.eval_fn is not None:
            try:
                eval_result = self.eval_fn(
                    int(server_round), new_global_labels, None
                )
                if isinstance(eval_result, dict):
                    for k, v in eval_result.items():
                        agg_metrics[f"server_eval_{k}"] = v
            except NotImplementedError:
                logger.debug(
                    "[Round %d] server eval stub (CNN not yet implemented)",
                    server_round,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("[Round %d] server eval failed: %s", server_round, e)

        self.round_metrics.append(dict(agg_metrics))
        logger.info(
            "[Round %d] aggregated: valid_global=%d familiar=%d unfamiliar=%d",
            server_round,
            agg_metrics["valid_global_labels"],
            agg_metrics["total_familiar"],
            agg_metrics["total_unfamiliar"],
        )
        return (
            ndarrays_to_parameters([self.global_labels.copy()]),
            agg_metrics,
        )

    # ---------- 7.5 ----------
    def vote_mechanism(
        self, all_client_labels: List[np.ndarray]
    ) -> np.ndarray:
        """Per-sample majority voting over non-(-1) predictions.

        Vectorized via `np.add.at` on a `(N_open, num_classes)` accumulator.
        """
        global_labels: np.ndarray = np.full(
            (self.n_open_samples,), -1, dtype=np.int64
        )
        voting_sets: np.ndarray = np.zeros(
            (self.n_open_samples, self.num_classes), dtype=np.int64
        )

        for client_labels in all_client_labels:
            if client_labels.shape[0] != self.n_open_samples:
                logger.warning(
                    "vote_mechanism: client label array length %d != N_open %d; skipping",
                    client_labels.shape[0],
                    self.n_open_samples,
                )
                continue
            valid_mask: np.ndarray = client_labels != -1
            valid_indices: np.ndarray = np.where(valid_mask)[0]
            valid_classes: np.ndarray = client_labels[valid_mask].astype(np.int64)
            # Defensive clamp for malformed uploads.
            valid_classes = np.clip(valid_classes, 0, self.num_classes - 1)
            np.add.at(voting_sets, (valid_indices, valid_classes), 1)

        total_votes: np.ndarray = voting_sets.sum(axis=1)
        has_votes: np.ndarray = total_votes > 0
        winning_classes: np.ndarray = voting_sets.argmax(axis=1)
        global_labels[has_votes] = winning_classes[has_votes]
        return global_labels

    # ---------- 7.6 ----------
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        eval_config: Dict[str, Scalar] = {"round": int(server_round)}
        eval_params: Parameters = ndarrays_to_parameters(
            [self.global_labels.copy()]
        )
        evaluate_ins = EvaluateIns(eval_params, eval_config)
        clients: List[ClientProxy] = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, evaluate_ins) for client in clients]

    # ---------- 7.7 ----------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        total_samples: int = sum(int(r.num_examples) for _, r in results)
        if total_samples == 0:
            return None, {}
        weighted_loss: float = float(
            sum(float(r.loss) * int(r.num_examples) for _, r in results)
        )
        weighted_accuracy: float = float(
            sum(
                float(r.metrics.get("accuracy", 0.0)) * int(r.num_examples)
                for _, r in results
            )
        )
        loss_agg: float = weighted_loss / total_samples
        accuracy_agg: float = weighted_accuracy / total_samples
        return loss_agg, {
            "accuracy": accuracy_agg,
            "round": int(server_round),
        }

    # ---------- Flower's optional centralized evaluate ----------
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Not used: server-side evaluation is folded into aggregate_fit via eval_fn."""
        return None
