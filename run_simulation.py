"""
SSFL Simulation Runner — Direct Execution (no Ray).

Runs the SSFL federated learning experiment by directly orchestrating
clients in-process. This avoids Ray entirely, eliminating common
macOS Ray runtime env issues while providing full control over
the simulation loop and progress display.

Usage:
    uv run python run_simulation.py                            # Default: Scenario 1, 200 rounds
    uv run python run_simulation.py --scenario 2 --rounds 50   # Custom
    uv run python run_simulation.py --scenario 1 --rounds 3    # Quick test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from src.data import get_num_clients, get_num_open_samples, load_split
from src.client_app import SSFLClient
from src.train import evaluate_model
from src.utils import compute_metrics, get_device, set_seed

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Progress display
# ─────────────────────────────────────────────────────────────────


def _format_time(seconds: float) -> str:
    """Format seconds into a readable time string."""
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{secs:02d}s"
    elif mins > 0:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def print_round_progress(
    server_round: int,
    num_rounds: int,
    num_clients: int,
    num_open: int,
    metrics: dict,
    best_acc: float,
    best_round: int,
    start_time: float,
) -> None:
    """Print a formatted progress panel after each round."""
    elapsed = time.time() - start_time
    pct = server_round / num_rounds * 100

    # Progress bar (30 chars wide)
    filled = int(30 * server_round / num_rounds)
    bar = "█" * filled + "░" * (30 - filled)

    # ETA
    if server_round > 0:
        per_round = elapsed / server_round
        remaining = per_round * (num_rounds - server_round)
        eta_str = _format_time(remaining)
    else:
        eta_str = "--"

    acc = metrics.get("test_accuracy", 0.0)
    f1 = metrics.get("test_f1", 0.0)
    cls_loss = metrics.get("avg_cls_loss", 0.0)
    dist_loss = metrics.get("avg_dist_loss", 0.0)
    labelled = int(metrics.get("num_labelled", 0))

    print(
        f"\n{'─' * 72}\n"
        f"  Round {server_round}/{num_rounds}  "
        f"|{bar}|  {pct:5.1f}%\n"
        f"{'─' * 72}\n"
        f"  Clients:   {num_clients}          "
        f"Labelled:  {labelled}/{num_open}\n"
        f"  Accuracy:  {acc:.4f}        "
        f"F1 Score:  {f1:.4f}\n"
        f"  Cls Loss:  {cls_loss:.4f}        "
        f"Dist Loss: {dist_loss:.4f}\n"
        f"  Elapsed:   {_format_time(elapsed)}          "
        f"ETA:       {eta_str}\n"
        f"  Best:      {best_acc:.4f} acc  "
        f"@ round {best_round}\n"
        f"{'─' * 72}",
        flush=True,
    )


def print_final_summary(
    scenario: int,
    num_rounds: int,
    best_acc: float,
    best_f1: float,
    best_round: int,
    last_metrics: dict,
    start_time: float,
) -> None:
    """Print a summary table after all rounds."""
    elapsed = time.time() - start_time
    print(
        f"\n{'═' * 72}\n"
        f"  ✅  SSFL SIMULATION COMPLETE\n"
        f"{'═' * 72}\n"
        f"  Scenario:     {scenario}\n"
        f"  Total rounds: {num_rounds}\n"
        f"  Total time:   {_format_time(elapsed)}\n"
        f"{'─' * 72}\n"
        f"  BEST RESULTS (round {best_round}):\n"
        f"    Accuracy:   {best_acc:.4f}\n"
        f"    F1 Score:   {best_f1:.4f}\n"
        f"{'─' * 72}\n"
        f"  LAST ROUND METRICS:",
        flush=True,
    )
    for key, value in sorted(last_metrics.items()):
        print(f"    {key:30s}  {value:.4f}", flush=True)
    print(f"{'═' * 72}\n", flush=True)


# ─────────────────────────────────────────────────────────────────
# Voting mechanism (server-side logic)
# ─────────────────────────────────────────────────────────────────


def majority_vote(
    all_predictions: list[np.ndarray],
    num_open_samples: int,
    num_classes: int = 11,
) -> np.ndarray:
    """
    Step 4 — Vote & Broadcast (Eq. 17).

    For each open sample j:
    - Collect all non-(-1) predictions from participating clients
    - Majority vote → global hard label
    """
    predictions_matrix = np.stack(all_predictions, axis=0)  # (K, N_open)
    global_labels = np.full(num_open_samples, -1, dtype=np.int16)

    for j in range(num_open_samples):
        votes = predictions_matrix[:, j]
        valid_votes = votes[votes >= 0]

        if len(valid_votes) == 0:
            continue

        vote_counts = np.bincount(valid_votes, minlength=num_classes)
        global_labels[j] = np.argmax(vote_counts)

    return global_labels


# ─────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SSFL federated learning simulation."
    )
    parser.add_argument(
        "--scenario", type=int, default=1, choices=[1, 2, 3],
        help="Non-IID data scenario (1=27, 2=89, 3=89 Dirichlet). Default: 1",
    )
    parser.add_argument(
        "--rounds", type=int, default=200,
        help="Number of communication rounds T. Default: 200",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate. Default: 0.0001",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size. Default: 100",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=5,
        help="Local training epochs per round. Default: 5",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    num_clients = get_num_clients(args.scenario)
    num_open = get_num_open_samples()

    # ── Startup banner ──
    scenario_desc = {
        1: "27 clients, shard-based",
        2: "89 clients, shard-based",
        3: "89 clients, Dirichlet(α=0.1)",
    }
    print(
        f"\n{'═' * 72}\n"
        f"  🌸  SSFL Simulation — Direct Execution\n"
        f"{'═' * 72}\n"
        f"  Scenario:      {args.scenario}  ({scenario_desc[args.scenario]})\n"
        f"  Num clients:   {num_clients}\n"
        f"  Num rounds:    {args.rounds}\n"
        f"  Learning rate: {args.lr}\n"
        f"  Batch size:    {args.batch_size}\n"
        f"  Local epochs:  {args.local_epochs}\n"
        f"  Seed:          {args.seed}\n"
        f"  Device:        {device}\n"
        f"{'═' * 72}\n"
        f"  Initialising {num_clients} clients...",
        end="", flush=True,
    )

    # ── Create all clients ──
    clients: list[SSFLClient] = []
    for cid in range(num_clients):
        client = SSFLClient(
            client_id=cid,
            scenario=args.scenario,
            lr=args.lr,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        clients.append(client)

    print(
        f" done.\n"
        f"{'═' * 72}\n"
        f"  Progress will be displayed after each round.\n"
        f"{'═' * 72}\n",
        flush=True,
    )

    # ── Load test set for evaluation ──
    test_loader = load_split("test", batch_size=args.batch_size, shuffle=False)

    # ── Tracking ──
    global_labels: np.ndarray | None = None
    best_acc = 0.0
    best_f1 = 0.0
    best_round = 0
    history: list[dict] = []
    start_time = time.time()

    # ─── Main FL loop ───
    for rnd in range(1, args.rounds + 1):

        # === Client-side: each client runs one SSFL iteration ===
        all_predictions = []
        all_cls_loss = []
        all_disc_loss = []
        all_dist_loss = []

        for client in clients:
            # Package server → client message (global labels or empty)
            if global_labels is not None:
                params = [global_labels.astype(np.float64)]
            else:
                params = [np.array([], dtype=np.float64)]

            config = {"server_round": rnd}

            # Run client fit
            result_params, num_samples, metrics = client.fit(params, config)
            hard_labels = result_params[0].astype(np.int16)

            all_predictions.append(hard_labels)
            all_cls_loss.append(metrics["classifier_loss"])
            all_disc_loss.append(metrics["discriminator_loss"])
            all_dist_loss.append(metrics["distillation_loss"])

        # === Server-side: voting (Eq. 17) ===
        global_labels = majority_vote(all_predictions, num_open)
        num_labelled = int(np.sum(global_labels >= 0))

        # === Evaluate using first client's classifier ===
        loss, y_true, y_pred = evaluate_model(
            clients[0].classifier, test_loader, device
        )
        eval_metrics = compute_metrics(y_true, y_pred)

        # Build round metrics
        round_metrics = {
            "test_accuracy": eval_metrics["accuracy"],
            "test_f1": eval_metrics["f1"],
            "test_precision": eval_metrics["precision"],
            "test_recall": eval_metrics["recall"],
            "test_loss": loss,
            "avg_cls_loss": float(np.mean(all_cls_loss)),
            "avg_disc_loss": float(np.mean(all_disc_loss)),
            "avg_dist_loss": float(np.mean(all_dist_loss)),
            "num_labelled": num_labelled,
        }

        # Track best
        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]
            best_f1 = eval_metrics["f1"]
            best_round = rnd

        history.append({"round": rnd, **round_metrics})

        # === Print progress ===
        print_round_progress(
            server_round=rnd,
            num_rounds=args.rounds,
            num_clients=num_clients,
            num_open=num_open,
            metrics=round_metrics,
            best_acc=best_acc,
            best_round=best_round,
            start_time=start_time,
        )

    # ── Final summary ──
    print_final_summary(
        scenario=args.scenario,
        num_rounds=args.rounds,
        best_acc=best_acc,
        best_f1=best_f1,
        best_round=best_round,
        last_metrics=history[-1] if history else {},
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
