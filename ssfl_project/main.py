"""Entry point — launches server or one client process."""
import argparse
import json
import logging
import os
import random

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
from server import build_eval_fn, start_server
from strategy import SSFLStrategy
from utils import setup_logging

logger = logging.getLogger(__name__)


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

    # Persist history + round-by-round metrics for later plotting.
    os.makedirs("logs", exist_ok=True)
    history_path = os.path.join("logs", "history.json")
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
        logger.info("Saved history to %s", history_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Could not save history JSON: %s", e)


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
