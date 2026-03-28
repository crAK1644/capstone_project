"""
SSFL Simulation Runner.

Entry point for running the Semisupervised Federated Learning experiment
using Flower's simulation engine. Simulates K virtual clients on a single
machine, faithfully reproducing the paper's federated setup.

Usage:
    uv run python run_simulation.py                         # Default: Scenario 1, 200 rounds
    uv run python run_simulation.py --scenario 2 --rounds 50  # Custom
    uv run python run_simulation.py --scenario 1 --rounds 5   # Quick test

The simulation uses Ray under the hood to manage client resources.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import flwr as fl
from flwr.simulation import run_simulation

from src.client_app import app as client_app
from src.server_app import app as server_app
from src.data import get_num_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SSFL federated learning simulation."
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Non-IID data scenario (1=27 clients, 2=89, 3=89 Dirichlet). Default: 1",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of communication rounds T. Default: 200",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate. Default: 0.0001",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size. Default: 100",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
        help="Local training epochs per round. Default: 5",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default: 42"
    )
    args = parser.parse_args()

    num_clients = get_num_clients(args.scenario)

    log.info("=" * 60)
    log.info("SSFL Simulation")
    log.info("=" * 60)
    log.info("  Scenario:      %d", args.scenario)
    log.info("  Num clients:   %d", num_clients)
    log.info("  Num rounds:    %d", args.rounds)
    log.info("  Learning rate: %f", args.lr)
    log.info("  Batch size:    %d", args.batch_size)
    log.info("  Local epochs:  %d", args.local_epochs)
    log.info("  Seed:          %d", args.seed)
    log.info("=" * 60)

    # Flower run configuration — passed to both server_fn and client_fn
    run_config = {
        "scenario": str(args.scenario),
        "num-rounds": str(args.rounds),
        "learning-rate": str(args.lr),
        "batch-size": str(args.batch_size),
        "local-epochs": str(args.local_epochs),
        "seed": str(args.seed),
    }

    # Run the simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": 0.0,  # Set > 0 if GPU available
            }
        },
        run_config=run_config,
    )

    log.info("Simulation complete!")


if __name__ == "__main__":
    main()
