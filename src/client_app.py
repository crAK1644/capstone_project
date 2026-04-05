"""
Flower ClientApp for SSFL.

Each client performs one full SSFL iteration per Flower round:

    Round 0 (no global labels yet):
        1. Train classifier on private data
        2. Compute confidence scores on open data
        3. Build D^{k,d}, train discriminator
        4. Filter & predict → upload hard labels

    Round ≥ 1 (global labels available):
        0. Distillation: train classifier on open data with global labels
        1–4. Same as above

The client maintains its Classifier and Discriminator models across
rounds using Flower's ``Context.state`` for persistence.
"""

from __future__ import annotations

import io
import logging
from collections import OrderedDict

import numpy as np
import torch

import flwr
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

from .data import load_client_data, load_client_data_tensors, load_open_data_tensors
from .model import Classifier, Discriminator
from .train import (
    build_discriminator_dataset,
    compute_confidence_scores,
    distillation_train,
    filter_and_predict,
    train_classifier,
    train_discriminator,
)
from .utils import get_device, labels_to_bytes, set_seed

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Flower NumPyClient implementation
# ─────────────────────────────────────────────────────────────────


class SSFLClient(NumPyClient):
    """
    SSFL Flower client.

    Each instance represents one IoT device (client k) participating
    in the federated training process.
    """

    def __init__(
        self,
        client_id: int,
        scenario: int,
        lr: float = 0.0001,
        local_epochs: int = 5,
        discriminator_epochs: int = 5,
        distillation_epochs: int = 5,
        batch_size: int = 100,
        num_classes: int = 11,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.scenario = scenario
        self.lr = lr
        self.local_epochs = local_epochs
        self.discriminator_epochs = discriminator_epochs
        self.distillation_epochs = distillation_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes

        set_seed(seed + client_id)
        self.device = get_device()

        # Initialise models
        self.classifier = Classifier(num_classes=num_classes)
        self.discriminator = Discriminator()

        # Pre-load data (stays in memory across rounds)
        self.private_loader = load_client_data(
            scenario, client_id, batch_size=batch_size, shuffle=True
        )
        self.X_private, _ = load_client_data_tensors(scenario, client_id)
        self.X_open, _ = load_open_data_tensors(use_2d=True)

        log.debug(
            "Client %d initialised: %d private samples, %d open samples",
            client_id,
            len(self.X_private),
            len(self.X_open),
        )

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """
        Return the client's hard-label predictions as 'parameters'.

        In SSFL, we do NOT exchange model weights — we exchange the
        hard-label predictions on the open dataset. We repurpose
        Flower's parameters mechanism to carry this payload.
        """
        # This is called by the framework; we return empty on initial round
        # Actual predictions are sent via fit() return value
        return [np.array([], dtype=np.int16)]

    def fit(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[list[np.ndarray], int, dict]:
        """
        Execute one SSFL round from the client's perspective.

        Args:
            parameters: From server — global hard labels P^s (or empty on round 0).
            config: Round configuration from the server strategy.

        Returns:
            Tuple of (hard_label_predictions, num_samples, metrics_dict).
        """
        server_round = config.get("server_round", 0)

        # ── Step 5 (if round ≥ 1): Distillation with global labels ──
        if len(parameters) > 0 and len(parameters[0]) > 0:
            global_labels = parameters[0].astype(np.int16)
            if len(global_labels) == len(self.X_open):
                log.debug(
                    "Client %d, round %d: distillation training",
                    self.client_id,
                    server_round,
                )
                dist_loss = distillation_train(
                    model=self.classifier,
                    X_open=self.X_open,
                    global_labels=global_labels,
                    epochs=self.distillation_epochs,
                    lr=self.lr,
                    device=self.device,
                    batch_size=self.batch_size,
                )
            else:
                log.warning(
                    "Client %d: global labels length mismatch (%d vs %d)",
                    self.client_id,
                    len(global_labels),
                    len(self.X_open),
                )
                dist_loss = 0.0
        else:
            dist_loss = 0.0

        # ── Step 1: Train classifier on private data ──
        log.debug(
            "Client %d, round %d: training classifier",
            self.client_id,
            server_round,
        )
        cls_loss = train_classifier(
            model=self.classifier,
            dataloader=self.private_loader,
            epochs=self.local_epochs,
            lr=self.lr,
            device=self.device,
        )

        # ── Step 1 (cont.): Compute confidence scores on open data ──
        predictions, confidences = compute_confidence_scores(
            model=self.classifier,
            X_open=self.X_open,
            device=self.device,
            batch_size=self.batch_size,
        )

        # ── Step 2: Build D^{k,d} and train discriminator ──
        theta = float(np.median(confidences))  # Paper recommendation
        disc_loader = build_discriminator_dataset(
            confidences=confidences,
            X_open=self.X_open,
            X_private=self.X_private,
            theta=theta,
            batch_size=self.batch_size,
        )
        disc_loss = train_discriminator(
            model=self.discriminator,
            dataloader=disc_loader,
            epochs=self.discriminator_epochs,
            lr=self.lr,
            device=self.device,
        )

        # ── Step 3: Filter predictions and produce hard labels ──
        hard_labels = filter_and_predict(
            classifier=self.classifier,
            discriminator=self.discriminator,
            X_open=self.X_open,
            device=self.device,
            batch_size=self.batch_size,
        )

        # Return hard labels as "parameters" + metrics
        num_familiar = int(np.sum(hard_labels >= 0))
        metrics = {
            "classifier_loss": float(cls_loss),
            "discriminator_loss": float(disc_loss),
            "distillation_loss": float(dist_loss),
            "theta": theta,
            "num_familiar": num_familiar,
            "num_unfamiliar": int(len(hard_labels) - num_familiar),
        }

        log.debug(
            "Client %d, round %d: %d familiar / %d unfamiliar",
            self.client_id,
            server_round,
            num_familiar,
            len(hard_labels) - num_familiar,
        )

        return [hard_labels.astype(np.float64)], len(self.X_private), metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[float, int, dict]:
        """
        Evaluate is handled centrally on the server side.
        Clients don't need to evaluate in SSFL.
        """
        return 0.0, 0, {}


# ─────────────────────────────────────────────────────────────────
# Client factory function for Flower simulation
# ─────────────────────────────────────────────────────────────────


def client_fn(context: Context) -> SSFLClient:
    """
    Factory function that creates an SSFLClient for Flower simulation.

    The ``node_config`` provides the ``partition-id`` (client index).
    Training hyperparameters come from the shared ``RUNTIME_CONFIG`` dict.
    """
    from .config import RUNTIME_CONFIG as cfg

    client_id = int(context.node_config["partition-id"])

    scenario = int(cfg.get("scenario", 1))
    lr = float(cfg.get("learning-rate", 0.0001))
    local_epochs = int(cfg.get("local-epochs", 5))
    batch_size = int(cfg.get("batch-size", 100))
    seed = int(cfg.get("seed", 42))

    return SSFLClient(
        client_id=client_id,
        scenario=scenario,
        lr=lr,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )


# Flower ClientApp instance
app = ClientApp(client_fn=client_fn)
