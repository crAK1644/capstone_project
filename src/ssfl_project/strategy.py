"""Custom Flower Strategy: voting-based aggregation over hard labels.

Fully implemented — no CNN dependency. The voting mechanism is the core
server-side algorithm of SSFL and this module can be unit-tested in
isolation.

Instrumentation note (see SSFL_FLOWER_INFRASTRUCTURE_PLAN.md §14):
`aggregate_fit` also maintains a `CommCostLedger` and (if `eval_fn` is
provided) a per-round classification-metric snapshot so the final Table
II / III / IV analogues can be produced without re-running training.
"""
import logging
import time
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

import config
from metrics import (
    CommCostLedger,
    RoundCommCost,
    open_dataset_distribution_bytes,
    payload_bytes_packed,
    payload_bytes_wire,
)

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
        charge_open_dataset_round: int = 1,
    ) -> None:
        super().__init__()
        self.num_clients: int = num_clients
        self.num_classes: int = num_classes
        self.n_open_samples: int = n_open_samples
        self.min_fit_clients: int = min_fit_clients
        self.min_available_clients: int = min_available_clients
        self.eval_fn: Optional[Callable] = eval_fn
        self.classifier_epochs: int = classifier_epochs
        self.charge_open_dataset_round: int = charge_open_dataset_round

        # Initial global labels: all -1 (no consensus yet).
        self.global_labels: np.ndarray = np.full(
            (n_open_samples,), -1, dtype=np.int64
        )
        self.round_metrics: List[dict] = []
        # Communication-cost accumulator (see metrics.CommCostLedger).
        self.comm_cost_ledger: CommCostLedger = CommCostLedger()

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
        round_t0: float = time.perf_counter()
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

        vote_t0: float = time.perf_counter()
        new_global_labels: np.ndarray = self.vote_mechanism(all_client_labels)
        vote_sec: float = time.perf_counter() - vote_t0
        self.global_labels = new_global_labels

        # ---- Aggregate client training metrics ----
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

        # Wall-clock diagnostics (pure-python, no CNN dependency).
        fit_times: List[float] = [
            float(m.get("fit_wall_clock_sec", 0.0))
            for m in client_metrics
            if "fit_wall_clock_sec" in m
        ]
        agg_metrics["avg_client_fit_sec"] = (
            float(np.mean(fit_times)) if fit_times else 0.0
        )
        agg_metrics["max_client_fit_sec"] = (
            float(np.max(fit_times)) if fit_times else 0.0
        )
        agg_metrics["server_vote_sec"] = float(vote_sec)

        thresholds: List[float] = [
            float(m.get("confidence_threshold", 0.0))
            for m in client_metrics
            if "confidence_threshold" in m
        ]
        agg_metrics["avg_confidence_threshold"] = (
            float(np.mean(thresholds)) if thresholds else 0.0
        )

        # ---- Communication cost accounting (Table IV analogue) ----
        uploaded_wire_total: int = sum(
            int(m.get("bytes_upload_wire", 0)) for m in client_metrics
        )
        uploaded_packed_total: int = sum(
            int(m.get("bytes_upload_packed", 0)) for m in client_metrics
        )
        # Server-to-client broadcast of global labels is fanned out to every
        # participating client, so the on-wire broadcast cost is
        # `bytes_per_labels_array × num_clients`.
        broadcast_arr: np.ndarray = new_global_labels.astype(np.int64)
        broadcast_wire_one: int = payload_bytes_wire(broadcast_arr)
        broadcast_packed_one: int = payload_bytes_packed(broadcast_arr)
        broadcast_wire_total: int = broadcast_wire_one * len(results)
        broadcast_packed_total: int = broadcast_packed_one * len(results)

        # Charge open-dataset distribution exactly once (default: round 1).
        # `×num_clients` because the paper counts the fan-out across all
        # recipients; pass `charge_open_dataset_round=0` at construction to
        # disable this term entirely.
        open_bytes_this_round: int = 0
        if server_round == self.charge_open_dataset_round:
            open_bytes_this_round = (
                open_dataset_distribution_bytes(self.n_open_samples)
                * self.num_clients
            )

        self.comm_cost_ledger.record(
            RoundCommCost(
                round=int(server_round),
                uploaded_bytes_wire=int(uploaded_wire_total),
                uploaded_bytes_packed=int(uploaded_packed_total),
                downloaded_bytes_wire=int(broadcast_wire_total),
                downloaded_bytes_packed=int(broadcast_packed_total),
                open_dataset_bytes=int(open_bytes_this_round),
            )
        )
        agg_metrics["bytes_upload_wire_total"] = int(uploaded_wire_total)
        agg_metrics["bytes_upload_packed_total"] = int(uploaded_packed_total)
        agg_metrics["bytes_broadcast_wire_total"] = int(broadcast_wire_total)
        agg_metrics["bytes_broadcast_packed_total"] = int(broadcast_packed_total)
        agg_metrics["bytes_open_dataset_this_round"] = int(open_bytes_this_round)
        agg_metrics["cumulative_mb_wire"] = float(
            self.comm_cost_ledger.cumulative_mb_at(
                int(server_round), packed=False
            )
        )
        agg_metrics["cumulative_mb_packed"] = float(
            self.comm_cost_ledger.cumulative_mb_at(
                int(server_round), packed=True
            )
        )

        # ---- Optional server-side evaluation (eval_fn is stubbed until CNN lands) ----
        # Signature: eval_fn(round, global_labels, X_open_or_None) -> dict with
        # accuracy, f1_*, precision_*, recall_*, confusion_matrix, per-class lists.
        if self.eval_fn is not None:
            eval_t0: float = time.perf_counter()
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
            agg_metrics["server_eval_sec"] = float(time.perf_counter() - eval_t0)

        agg_metrics["round_total_sec"] = float(time.perf_counter() - round_t0)

        self.round_metrics.append(dict(agg_metrics))
        logger.info(
            "[Round %d] aggregated: valid_global=%d familiar=%d unfamiliar=%d "
            "upload=%.4f MB cum=%.3f MB",
            server_round,
            agg_metrics["valid_global_labels"],
            agg_metrics["total_familiar"],
            agg_metrics["total_unfamiliar"],
            float(uploaded_wire_total) / 1_000_000.0,
            agg_metrics["cumulative_mb_wire"],
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
