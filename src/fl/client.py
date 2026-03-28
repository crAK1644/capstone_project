from __future__ import annotations

from collections.abc import Callable

import torch
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar
from torch import nn

from src.data.loaders import make_client_loaders, make_open_loader
from src.models import IoTCNN
from src.utils.config import RunConfig
from src.utils.training import evaluate_classifier, train_local, train_open_set_consistency


def get_parameters(model: nn.Module) -> NDArrays:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    state_dict = model.state_dict()
    params_dict = zip(state_dict.keys(), parameters, strict=True)
    new_state = {key: torch.tensor(value) for key, value in params_dict}
    model.load_state_dict(new_state, strict=True)


class IoTNumPyClient(NumPyClient):
    def __init__(
        self,
        client_id: int,
        run_config: RunConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.run_config = run_config
        self.device = device
        self.model = IoTCNN(num_classes=11).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.run_config.local_epochs))
        train_loader, val_loader = make_client_loaders(
            scenario=self.run_config.scenario,
            client_id=self.client_id,
            batch_size=self.run_config.batch_size,
            val_ratio=self.run_config.val_ratio,
            seed=self.run_config.seed,
            use_2d=True,
            base_dir=self.run_config.prepared_data_dir,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.run_config.learning_rate,
            weight_decay=self.run_config.weight_decay,
        )

        train_metrics = train_local(
            model=self.model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=self.criterion,
            device=self.device,
            epochs=local_epochs,
        )

        ssfl_metrics: dict[str, float] = {"ssfl_loss": 0.0, "ssfl_acceptance": 0.0}
        if self.run_config.ssfl_enabled:
            open_loader = make_open_loader(
                batch_size=self.run_config.batch_size,
                use_2d=True,
                base_dir=self.run_config.prepared_data_dir,
            )
            ssfl_metrics = train_open_set_consistency(
                model=self.model,
                loader=open_loader,
                optimizer=optimizer,
                device=self.device,
                confidence_threshold=self.run_config.ssfl_confidence_threshold,
                weight=self.run_config.ssfl_lambda,
            )

        val_metrics = evaluate_classifier(
            model=self.model,
            loader=val_loader,
            criterion=self.criterion,
            device=self.device,
            num_classes=11,
        )

        metrics: dict[str, Scalar] = {
            "train_loss": float(train_metrics["train_loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "ssfl_loss": float(ssfl_metrics["ssfl_loss"]),
            "ssfl_acceptance": float(ssfl_metrics["ssfl_acceptance"]),
        }

        num_examples = sum(int(yb.size(0)) for _, yb in train_loader)
        return get_parameters(self.model), num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        set_parameters(self.model, parameters)

        _, val_loader = make_client_loaders(
            scenario=self.run_config.scenario,
            client_id=self.client_id,
            batch_size=self.run_config.batch_size,
            val_ratio=self.run_config.val_ratio,
            seed=self.run_config.seed,
            use_2d=True,
            base_dir=self.run_config.prepared_data_dir,
        )

        val_metrics = evaluate_classifier(
            model=self.model,
            loader=val_loader,
            criterion=self.criterion,
            device=self.device,
            num_classes=11,
        )

        return (
            float(val_metrics["loss"]),
            int(val_metrics["num_examples"]),
            {
                "accuracy": float(val_metrics["accuracy"]),
                "macro_f1": float(val_metrics["macro_f1"]),
            },
        )


def make_client_fn(
    client_ids: list[int],
    run_config: RunConfig,
) -> Callable[[str], Client]:
    def client_fn(cid: str) -> Client:
        index = int(cid)
        client_id = client_ids[index]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return IoTNumPyClient(client_id=client_id, run_config=run_config, device=device).to_client()

    return client_fn
