"""Entry point — launches server or one client process.

Metrics wiring (see SSFL_FLOWER_INFRASTRUCTURE_PLAN.md §14):
  On server shutdown, `run_server` persists four artefacts under
  `args.metrics_dir` (default `metrics/`):
      per_round.json        — full round_metrics history, all keys, list-shaped.
      per_round.csv         — scalar-only view of the same history (1 row = 1 round).
      summary.json          — `metrics.build_summary_report` output (Tables II-IV cells).
      confusion_matrix_final.json — final-round confusion matrix + per-class F1/P/R/support.
  A `history.json` is also written under `args.logs_dir` (default `logs/`) for
  Flower-native fields (losses_distributed / metrics_distributed{_fit}).
"""
import argparse
import json
import logging
import os
import random
from typing import Tuple

import flwr as fl
import numpy as np
import torch

import config
from client import SSFLClient
from data_preparation import (
    load_client_partition,
    load_open_data,
    load_test_data,
)
from metrics import (
    build_summary_report,
    save_confusion_matrix_json,
    save_metrics_csv,
    save_metrics_json,
    save_summary_json,
)
from server import build_eval_fn, start_server
from strategy import SSFLStrategy
from utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_float_tuple(s: str) -> Tuple[float, ...]:
    """CLI helper: '0.50,0.75' -> (0.50, 0.75)."""
    if not s:
        return ()
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    """CLI helper: '10,50,100' -> (10, 50, 100)."""
    if not s:
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


# ---------- 9.1 ----------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSFL Flower entry point")
    parser.add_argument(
        "--mode", choices=["server", "client"], required=True,
        help="Whether this process plays the server role or a single client role.",
    )
    parser.add_argument(
        "--client_id", type=int, default=0,
        help="Global client ID (0..NUM_CLIENTS-1); client mode only.",
    )
    parser.add_argument(
        "--server_address", type=str, default=config.DEFAULT_SERVER_ADDRESS,
    )
    parser.add_argument(
        "--num_rounds", type=int, default=config.NUM_ROUNDS,
        help="Number of SSFL communication rounds (server mode only).",
    )
    parser.add_argument(
        "--num_clients", type=int, default=config.NUM_CLIENTS,
        help="Total number of clients participating.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=config.NUM_CLASSES,
        help="Size of the global label space.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=config.LEARNING_RATE,
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--classifier_epochs", type=int, default=config.CLASSIFIER_EPOCHS,
    )
    parser.add_argument(
        "--discriminator_epochs", type=int, default=config.DISCRIMINATOR_EPOCHS,
    )
    parser.add_argument(
        "--partition_dir", type=str, default=config.PARTITION_DIR,
    )
    parser.add_argument(
        "--data_dir", type=str, default=config.DATA_DIR,
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
    )
    # -------- Metrics / evaluation protocol (§14) --------
    parser.add_argument(
        "--metrics_dir", type=str, default=config.METRICS_DIR,
        help="Directory where per_round.{json,csv}, summary.json, and "
             "confusion_matrix_final.json are written (server mode only).",
    )
    parser.add_argument(
        "--logs_dir", type=str, default=config.LOGS_DIR,
        help="Directory where Flower history.json is written.",
    )
    parser.add_argument(
        "--target_accs", type=str,
        default=",".join(f"{a:.2f}" for a in config.TARGET_ACCURACIES),
        help="Comma-separated target accuracies for C@x (Table IV). "
             "Default: '0.50,0.75'.",
    )
    parser.add_argument(
        "--snapshot_rounds", type=str,
        default=",".join(str(r) for r in config.SNAPSHOT_ROUNDS),
        help="Comma-separated rounds at which to snapshot Top-1 accuracy "
             "(Table III). Default: '10,50,100,150,200'.",
    )
    return parser.parse_args()


# ---------- 9.2 ----------
def setup_environment(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    setup_logging()


# ---------- 9.3 ----------
def run_server(args: argparse.Namespace) -> None:
    device: torch.device = torch.device(args.device)

    X_test, y_test = load_test_data(args.partition_dir)
    X_open: np.ndarray = load_open_data(args.partition_dir)
    n_open_samples: int = int(X_open.shape[0])

    target_accs: Tuple[float, ...] = _parse_float_tuple(args.target_accs)
    snapshot_rounds: Tuple[int, ...] = _parse_int_tuple(args.snapshot_rounds)

    # eval_fn closure is built now; its body is stubbed until CNN lands.
    eval_fn = build_eval_fn(X_test, y_test, args.num_classes, device)

    strategy = SSFLStrategy(
        num_clients=args.num_clients,
        num_classes=args.num_classes,
        n_open_samples=n_open_samples,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        eval_fn=eval_fn,
        classifier_epochs=args.classifier_epochs,
    )
    history = start_server(args.server_address, strategy, args.num_rounds)

    # --- Flower-native history (unchanged — other tooling may depend on it) ---
    os.makedirs(args.logs_dir, exist_ok=True)
    history_path: str = os.path.join(args.logs_dir, "history.json")
    try:
        with open(history_path, "w") as f:
            json.dump(
                {
                    "losses_distributed": history.losses_distributed,
                    "metrics_distributed_fit": history.metrics_distributed_fit,
                    "metrics_distributed": history.metrics_distributed,
                    "round_metrics": strategy.round_metrics,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info("Saved Flower history to %s", history_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Could not save history JSON: %s", e)

    # --- §14 deliverables: per-round, summary, final confusion matrix ---
    os.makedirs(args.metrics_dir, exist_ok=True)
    per_round_json: str = os.path.join(args.metrics_dir, "per_round.json")
    per_round_csv: str = os.path.join(args.metrics_dir, "per_round.csv")
    summary_json: str = os.path.join(args.metrics_dir, "summary.json")
    cm_json: str = os.path.join(args.metrics_dir, "confusion_matrix_final.json")

    try:
        save_metrics_json(strategy.round_metrics, per_round_json)
        save_metrics_csv(strategy.round_metrics, per_round_csv)
        logger.info("Saved per-round metrics to %s / %s", per_round_json, per_round_csv)
    except Exception as e:  # pragma: no cover
        logger.warning("Could not save per-round metrics: %s", e)

    try:
        summary: dict = build_summary_report(
            strategy.round_metrics,
            strategy.comm_cost_ledger,
            target_accs=target_accs or config.TARGET_ACCURACIES,
            snapshot_rounds=snapshot_rounds or config.SNAPSHOT_ROUNDS,
        )
        # Attach the ledger in its entirety so plotting code doesn't have to
        # reconstruct cumulative byte series from the per-round scalars.
        summary["comm_cost_ledger"] = strategy.comm_cost_ledger.to_dict()
        save_summary_json(summary, summary_json)
        logger.info(
            "Saved summary (top_acc=%.4f, C@top_acc_packed=%s MB) to %s",
            float(summary.get("top_acc", 0.0)),
            summary.get("c_at_top_acc_packed_mb"),
            summary_json,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("Could not build/save summary JSON: %s", e)

    # Final-round confusion matrix: pull the last server_eval_* block out of
    # round_metrics and persist it for the Fig. 3 heatmap. If eval_fn was still
    # stubbed for the entire run, we simply skip this file.
    try:
        final_cm_payload: dict = {}
        for row in reversed(strategy.round_metrics):
            if "server_eval_confusion_matrix" in row:
                final_cm_payload = {
                    "confusion_matrix": row.get("server_eval_confusion_matrix"),
                    "class_names": row.get("server_eval_class_names"),
                    "f1_per_class": row.get("server_eval_f1_per_class"),
                    "precision_per_class": row.get("server_eval_precision_per_class"),
                    "recall_per_class": row.get("server_eval_recall_per_class"),
                    "support_per_class": row.get("server_eval_support_per_class"),
                }
                break
        if final_cm_payload:
            save_confusion_matrix_json(final_cm_payload, cm_json)
            logger.info("Saved final confusion matrix to %s", cm_json)
        else:
            logger.info(
                "No server_eval_confusion_matrix found in round_metrics "
                "(eval_fn likely stubbed); skipping %s", cm_json,
            )
    except Exception as e:  # pragma: no cover
        logger.warning("Could not save final confusion matrix: %s", e)


# ---------- 9.4 ----------
def run_client(args: argparse.Namespace) -> None:
    device: torch.device = torch.device(args.device)

    X_private, y_private = load_client_partition(args.client_id, args.partition_dir)
    X_open: np.ndarray = load_open_data(args.partition_dir)

    ssfl_client = SSFLClient(
        client_id=args.client_id,
        X_private=X_private,
        y_private=y_private,
        X_open=X_open,
        num_classes=args.num_classes,
        device=device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        classifier_epochs=args.classifier_epochs,
        discriminator_epochs=args.discriminator_epochs,
    )

    # Flower >= 1.4: use start_client + to_client() instead of deprecated
    # start_numpy_client (see plan §9.4 corrected note).
    fl.client.start_client(
        server_address=args.server_address,
        client=ssfl_client.to_client(),
    )


# ---------- 9.5 ----------
def main() -> None:
    args = parse_arguments()
    setup_environment(args.seed)
    if args.mode == "server":
        run_server(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()
