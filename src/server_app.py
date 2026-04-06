"""
Flower ServerApp and custom SSFLStrategy for SSFL.

The SSFL strategy is fundamentally different from FedAvg:
- Clients upload **hard-label predictions** on the shared open dataset
  (not model parameters).
- The server aggregates via **majority voting** (Eq. 17), not weighted
  parameter averaging.
- The aggregated global hard labels P^s are broadcast back to all clients
  for distillation in the next round.

Additionally, a server-side global model is trained on the open data with
global labels (Eq. 10) and used for centralised evaluation on the test set.
"""

from __future__ import annotations

import logging
import time
from typing import Union

import numpy as np

import flwr
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from .data import get_num_clients, get_num_open_samples, load_split
from .model import Classifier
from .train import evaluate_model
from .utils import compute_metrics, get_device, set_seed

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Custom SSFL Strategy
# ─────────────────────────────────────────────────────────────────


class SSFLStrategy(Strategy):
    """
    Custom Flower strategy implementing SSFL's server-side logic.

    Key differences from FedAvg:
    1. ``configure_fit()`` sends global hard labels (not model weights)
    2. ``aggregate_fit()`` uses majority voting (not weighted averaging)
    3. The server maintains its own global classifier for evaluation
    """

    def __init__(
        self,
        *,
        num_rounds: int = 200,
        num_classes: int = 11,
        num_open_samples: int | None = None,
        scenario: int = 1,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        lr: float = 0.0001,
        seed: int = 42,
    ) -> None:
        """
        Args:
            num_rounds: Total communication rounds T.
            num_classes: Number of traffic categories (L = 11).
            num_open_samples: Size of the open dataset. Auto-detected if None.
            scenario: Which non-IID scenario (1, 2, or 3).
            fraction_fit: Fraction of available clients to sample each round.
            min_fit_clients: Minimum number of clients required for a round.
            min_available_clients: Minimum clients that must be connected.
            lr: Learning rate for the server's global model.
            seed: Random seed.
        """
        super().__init__()
        self.num_rounds = num_rounds
        self.num_classes = num_classes
        self.scenario = scenario
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.lr = lr

        set_seed(seed)
        self.device = get_device()

        # Auto-detect open dataset size
        self.num_open_samples = (
            num_open_samples if num_open_samples else get_num_open_samples()
        )

        # Server's global hard labels (initialised empty — no labels on round 0)
        self.global_labels: np.ndarray | None = None

        # Server-side global classifier for evaluation (Eq. 10)
        self.global_model = Classifier(num_classes=num_classes)

        # Metrics history for plotting
        self.history: list[dict[str, float]] = []

        # Progress tracking
        self._start_time: float | None = None
        self._best_accuracy: float = 0.0
        self._best_f1: float = 0.0
        self._best_round: int = 0

        log.debug(
            "SSFLStrategy initialised: scenario=%d, open_samples=%d, classes=%d",
            scenario,
            self.num_open_samples,
            num_classes,
        )

    # ─── Client sampling ───────────────────────────────────────

    def initialize_parameters(self, client_manager) -> Parameters | None:
        """No initial parameters — SSFL doesn't share model weights."""
        return None

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Determine how many clients to sample for fit."""
        num_clients = max(
            int(num_available_clients * self.fraction_fit),
            self.min_fit_clients,
        )
        return num_clients, self.min_available_clients

    def num_evaluate_clients(self, num_available_clients: int) -> tuple[int, int]:
        """No client-side evaluation in SSFL."""
        return 0, 0

    # ─── Configure fit (send global labels to clients) ─────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Send global hard labels P^s to all sampled clients.

        - Round 1: send empty array (no global labels yet)
        - Round ≥ 2: send the P^s computed in the previous round's aggregation
        """
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        # Prepare global labels to send
        if self.global_labels is not None:
            params = ndarrays_to_parameters([self.global_labels.astype(np.float64)])
        else:
            params = ndarrays_to_parameters([np.array([], dtype=np.float64)])

        config = {"server_round": server_round}
        fit_ins = FitIns(parameters=params, config=config)

        return [(client, fit_ins) for client in clients]

    # ─── Aggregate fit: voting mechanism (Eq. 17) ──────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Step 4 — Vote & Broadcast.

        Aggregates hard-label predictions from all clients via majority
        voting (Eq. 17) to produce global hard labels P^s.

        For each open sample j:
        - Collect all non-(-1) predictions from participating clients
        - Count votes per class in voting sets V_{j,0}, ..., V_{j,L-1}
        - Global label = argmax of vote counts

        If no client provides a valid prediction for a sample, the global
        label defaults to -1 (will be skipped in distillation).
        """
        # Start timer on first round
        if self._start_time is None:
            self._start_time = time.time()

        if not results:
            log.warning("Round %d: no results received!", server_round)
            return None, {}

        # Collect hard labels from all clients
        all_predictions = []
        client_metrics = {}
        for client_proxy, fit_res in results:
            client_labels = parameters_to_ndarrays(fit_res.parameters)[0]
            all_predictions.append(client_labels.astype(np.int16))

            # Aggregate per-client metrics
            for key, value in fit_res.metrics.items():
                if key not in client_metrics:
                    client_metrics[key] = []
                client_metrics[key].append(value)

        # --- Majority voting (Eq. 17) ---
        predictions_matrix = np.stack(all_predictions, axis=0)  # (K, N_open)
        global_labels = np.full(self.num_open_samples, -1, dtype=np.int16)

        for j in range(self.num_open_samples):
            votes = predictions_matrix[:, j]
            valid_votes = votes[votes >= 0]

            if len(valid_votes) == 0:
                continue  # No client claimed this sample as familiar

            # Count votes per class
            vote_counts = np.bincount(valid_votes, minlength=self.num_classes)
            global_labels[j] = np.argmax(vote_counts)

        self.global_labels = global_labels

        # --- Compute aggregation statistics ---
        num_labelled = int(np.sum(global_labels >= 0))
        num_unlabelled = int(self.num_open_samples - num_labelled)

        # Average client metrics
        avg_metrics = {
            f"avg_{key}": float(np.mean(values))
            for key, values in client_metrics.items()
        }
        avg_metrics["num_labelled_open"] = float(num_labelled)
        avg_metrics["num_unlabelled_open"] = float(num_unlabelled)

        # --- Evaluate server's global model on test set ---
        eval_metrics = self._evaluate_global_model(server_round)
        avg_metrics.update(eval_metrics)

        # Store history for plotting
        self.history.append({"round": server_round, **avg_metrics})

        # --- Print progress ---
        self._print_round_progress(server_round, avg_metrics, len(results))

        # Print final summary on last round
        if server_round == self.num_rounds:
            self._print_final_summary()

        # Return None for parameters (no model weights to aggregate)
        return None, avg_metrics

    # ─── Server-side evaluation ────────────────────────────────

    def _evaluate_global_model(self, server_round: int) -> dict[str, float]:
        """
        Evaluate the server's global classifier on the test set.

        Note: In a full implementation, the server model would be trained
        via Eq. 10. For simplicity in the initial skeleton, we evaluate
        using the first available client model. This should be extended
        in Phase 3.
        """
        try:
            test_loader = load_split("test", batch_size=100, shuffle=False)
            loss, y_true, y_pred = evaluate_model(
                self.global_model, test_loader, self.device
            )
            metrics = compute_metrics(y_true, y_pred, self.num_classes)

            log.debug(
                "Round %d eval — Accuracy: %.4f, F1: %.4f, Loss: %.4f",
                server_round,
                metrics["accuracy"],
                metrics["f1"],
                loss,
            )

            return {
                "test_loss": loss,
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
            }
        except Exception as e:
            log.error("Evaluation failed: %s", e)
            return {}

    # ─── Progress display ──────────────────────────────────────

    def _print_round_progress(
        self,
        server_round: int,
        metrics: dict[str, float],
        num_clients: int,
    ) -> None:
        """Print a formatted progress line after each round."""
        elapsed = time.time() - (self._start_time or time.time())
        pct = server_round / self.num_rounds * 100

        # Progress bar (30 chars wide)
        filled = int(30 * server_round / self.num_rounds)
        bar = "█" * filled + "░" * (30 - filled)

        # Extract key metrics (with fallbacks)
        acc = metrics.get("test_accuracy", 0.0)
        f1 = metrics.get("test_f1", 0.0)
        cls_loss = metrics.get("avg_classifier_loss", 0.0)
        dist_loss = metrics.get("avg_distillation_loss", 0.0)
        labelled = int(metrics.get("num_labelled_open", 0))

        # Track best
        if acc > self._best_accuracy:
            self._best_accuracy = acc
            self._best_f1 = f1
            self._best_round = server_round

        # Format elapsed time
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            time_str = f"{hrs}h{mins:02d}m{secs:02d}s"
        elif mins > 0:
            time_str = f"{mins}m{secs:02d}s"
        else:
            time_str = f"{secs}s"

        # ETA
        if server_round > 0:
            per_round = elapsed / server_round
            remaining = per_round * (self.num_rounds - server_round)
            eta_mins, eta_secs = divmod(int(remaining), 60)
            eta_hrs, eta_mins = divmod(eta_mins, 60)
            if eta_hrs > 0:
                eta_str = f"{eta_hrs}h{eta_mins:02d}m"
            elif eta_mins > 0:
                eta_str = f"{eta_mins}m{eta_secs:02d}s"
            else:
                eta_str = f"{eta_secs}s"
        else:
            eta_str = "--"

        # ── Print the progress line ──
        print(
            f"\n{'─' * 72}\n"
            f"  Round {server_round}/{self.num_rounds}  "
            f"|{bar}|  {pct:5.1f}%\n"
            f"{'─' * 72}\n"
            f"  Clients:   {num_clients}    "
            f"Labelled:  {labelled}/{self.num_open_samples}\n"
            f"  Accuracy:  {acc:.4f}        "
            f"F1 Score:  {f1:.4f}\n"
            f"  Cls Loss:  {cls_loss:.4f}        "
            f"Dist Loss: {dist_loss:.4f}\n"
            f"  Elapsed:   {time_str}          "
            f"ETA:       {eta_str}\n"
            f"  Best:      {self._best_accuracy:.4f} acc  "
            f"@ round {self._best_round}\n"
            f"{'─' * 72}",
            flush=True,
        )

    def _print_final_summary(self) -> None:
        """Print a summary table after all rounds complete."""
        elapsed = time.time() - (self._start_time or time.time())
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            time_str = f"{hrs}h {mins:02d}m {secs:02d}s"
        elif mins > 0:
            time_str = f"{mins}m {secs:02d}s"
        else:
            time_str = f"{secs}s"

        print(
            f"\n{'═' * 72}\n"
            f"  ✅  SSFL SIMULATION COMPLETE\n"
            f"{'═' * 72}\n"
            f"  Scenario:     {self.scenario}\n"
            f"  Total rounds: {self.num_rounds}\n"
            f"  Total time:   {time_str}\n"
            f"{'─' * 72}\n"
            f"  BEST RESULTS (round {self._best_round}):\n"
            f"    Accuracy:   {self._best_accuracy:.4f}\n"
            f"    F1 Score:   {self._best_f1:.4f}\n"
            f"{'─' * 72}\n"
            f"  LAST ROUND METRICS:",
            flush=True,
        )
        if self.history:
            last = self.history[-1]
            for key, value in sorted(last.items()):
                if key == "round":
                    continue
                print(f"    {key:30s}  {value:.4f}", flush=True)
        print(f"{'═' * 72}\n", flush=True)

    # ─── Unused methods (required by Strategy interface) ───────

    def configure_evaluate(self, server_round, parameters, client_manager):
        """No client-side evaluation in SSFL."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """No client-side evaluation to aggregate."""
        return None, {}

    def evaluate(self, server_round, parameters):
        """Server-side evaluation is done in aggregate_fit."""
        return None


# ─────────────────────────────────────────────────────────────────
# ServerApp factory
# ─────────────────────────────────────────────────────────────────


def server_fn(context: flwr.common.Context) -> ServerAppComponents:
    """
    Factory function that creates the ServerApp components.

    Reads configuration from the shared ``RUNTIME_CONFIG`` dict
    (set by ``run_simulation.py`` before the simulation starts).
    """
    from .config import RUNTIME_CONFIG as cfg

    num_rounds = int(cfg.get("num-rounds", 200))
    scenario = int(cfg.get("scenario", 1))
    lr = float(cfg.get("learning-rate", 0.0001))
    seed = int(cfg.get("seed", 42))
    fraction_fit = float(cfg.get("fraction-fit", 1.0))

    num_clients = get_num_clients(scenario)

    strategy = SSFLStrategy(
        num_rounds=num_rounds,
        scenario=scenario,
        lr=lr,
        seed=seed,
        fraction_fit=fraction_fit,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp instance
app = ServerApp(server_fn=server_fn)

