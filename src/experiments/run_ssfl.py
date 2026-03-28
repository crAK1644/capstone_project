from __future__ import annotations

import argparse
from collections.abc import Sequence

from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from src.data.loaders import get_client_ids
from src.fl import build_fedavg_strategy, make_client_fn
from src.utils.config import RunConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flower FedAvg + SSFL open-set consistency.")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fraction-fit", type=float, default=0.3)
    parser.add_argument("--fraction-evaluate", type=float, default=0.3)
    parser.add_argument("--min-fit-clients", type=int, default=5)
    parser.add_argument("--min-evaluate-clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssfl-lambda", type=float, default=0.2)
    parser.add_argument("--ssfl-threshold", type=float, default=0.9)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    run_config = RunConfig(
        scenario=args.scenario,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=max(args.min_fit_clients, args.min_evaluate_clients),
        seed=args.seed,
        ssfl_enabled=True,
        ssfl_lambda=args.ssfl_lambda,
        ssfl_confidence_threshold=args.ssfl_threshold,
    )

    client_ids = get_client_ids(run_config.scenario, base_dir=run_config.prepared_data_dir)
    strategy = build_fedavg_strategy(run_config)

    history = start_simulation(
        client_fn=make_client_fn(client_ids=client_ids, run_config=run_config),
        num_clients=len(client_ids),
        config=ServerConfig(num_rounds=run_config.rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": run_config.num_cpus_per_client,
            "num_gpus": run_config.num_gpus_per_client,
        },
    )

    print("Training completed.")
    print(history)


if __name__ == "__main__":
    main()
