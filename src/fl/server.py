from __future__ import annotations

from collections.abc import Callable

import torch
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedAvg
from torch import nn

from src.data.loaders import make_test_loader
from src.fl.client import set_parameters
from src.models import IoTCNN
from src.utils.config import RunConfig
from src.utils.training import evaluate_classifier


def make_fit_config_fn(run_config: RunConfig) -> Callable[[int], dict[str, Scalar]]:
    def fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": run_config.local_epochs,
        }

    return fit_config


def make_server_eval_fn(run_config: RunConfig) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]]]:
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = make_test_loader(
        batch_size=run_config.batch_size,
        use_2d=True,
        base_dir=run_config.prepared_data_dir,
    )

    def evaluate(server_round: int, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, dict[str, Scalar]]:
        model = IoTCNN(num_classes=11).to(device)
        set_parameters(model, parameters)
        metrics = evaluate_classifier(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=11,
        )
        return float(metrics["loss"]), {
            "test_accuracy": float(metrics["accuracy"]),
            "test_macro_f1": float(metrics["macro_f1"]),
            "server_round": float(server_round),
        }

    return evaluate


def build_fedavg_strategy(run_config: RunConfig) -> FedAvg:
    return FedAvg(
        fraction_fit=run_config.fraction_fit,
        fraction_evaluate=run_config.fraction_evaluate,
        min_fit_clients=run_config.min_fit_clients,
        min_evaluate_clients=run_config.min_evaluate_clients,
        min_available_clients=run_config.min_available_clients,
        on_fit_config_fn=make_fit_config_fn(run_config),
        evaluate_fn=make_server_eval_fn(run_config),
    )
