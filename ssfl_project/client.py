"""Flower NumPyClient — SSFLClient orchestrates all 5 SSFL steps per round.

The orchestration (`fit`) and Flower-channel plumbing (`get_parameters`,
`set_parameters`) are fully wired. Per-step methods that require a real
CNN forward/backward (`train_classifier`, `compute_confidence_scores`,
`train_discriminator`, `filter_and_predict`, `run_distillation` beyond
the round-1 short-circuit, and `evaluate`) raise NotImplementedError until
model.py ships.
"""
import logging
from typing import List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from config import DISTILLATION_EPOCHS
from model import TrafficCNN, build_classifier, build_discriminator

logger = logging.getLogger(__name__)


class SSFLClient(fl.client.NumPyClient):
    """One Flower client per simulated IoT client (27 total in Scenario 1)."""

    # ---------- 6.1 Constructor ----------
    def __init__(
        self,
        client_id: int,
        X_private: np.ndarray,
        y_private: np.ndarray,
        X_open: np.ndarray,
        num_classes: int,
        device: torch.device,
        learning_rate: float,
        batch_size: int,
        classifier_epochs: int,
        discriminator_epochs: int,
        distillation_epochs: int = DISTILLATION_EPOCHS,
    ) -> None:
        self.client_id: int = client_id
        self.X_private: np.ndarray = X_private
        self.y_private: np.ndarray = y_private
        self.X_open: np.ndarray = X_open
        self.N_open: int = int(len(X_open))
        self.num_classes: int = num_classes
        self.device: torch.device = device
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.classifier_epochs: int = classifier_epochs
        self.discriminator_epochs: int = discriminator_epochs
        self.distillation_epochs: int = distillation_epochs

        # Model factories — will raise NotImplementedError until CNN lands.
        # This is intentional: model.py is the undeclared entry point.
        self.classifier: TrafficCNN = build_classifier(num_classes, device)
        self.discriminator: TrafficCNN = build_discriminator(device)
        self.classifier_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.classifier.parameters(), lr=learning_rate
        )

        self.current_round: int = 0
        self.global_labels: Optional[np.ndarray] = None

    # ---------- 6.2 Step 1a: classifier training ----------
    def train_classifier(self) -> float:
        """Train the classifier on private labeled data for self.classifier_epochs.

        STUB — depends on CNN forward/backward.
        """
        raise NotImplementedError(
            "train_classifier deferred: requires real TrafficCNN."
        )

    # ---------- 6.3 Step 1b: confidence scores ----------
    def compute_confidence_scores(self) -> np.ndarray:
        """Max-softmax confidence per open sample; shape (N_open,).

        STUB — depends on CNN forward.
        """
        raise NotImplementedError(
            "compute_confidence_scores deferred: requires real TrafficCNN."
        )

    # ---------- 6.4 Step 1c: adaptive median threshold ----------
    def compute_confidence_threshold(
        self, confidence_scores: np.ndarray
    ) -> float:
        """Adaptive threshold θ = median of this client's confidence scores."""
        return float(np.median(confidence_scores))

    # ---------- 6.5 Step 2a: discriminator dataset assembly ----------
    def build_discriminator_dataset(
        self, confidence_scores: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble familiar (private rows, label 0) and unfamiliar (low-confidence
        open rows, label 1) into a single training set for the discriminator.

        High-confidence open rows are deliberately excluded from the familiar
        side (see plan §6.5 design note).
        """
        unfamiliar_mask: np.ndarray = confidence_scores < threshold
        X_unfamiliar: np.ndarray = self.X_open[unfamiliar_mask]
        y_unfamiliar: np.ndarray = np.ones((len(X_unfamiliar),), dtype=np.int64)
        X_private_familiar: np.ndarray = self.X_private
        y_private_familiar: np.ndarray = np.zeros(
            (len(X_private_familiar),), dtype=np.int64
        )
        X_disc: np.ndarray = np.concatenate(
            [X_unfamiliar, X_private_familiar], axis=0
        )
        y_disc: np.ndarray = np.concatenate(
            [y_unfamiliar, y_private_familiar], axis=0
        )
        return X_disc, y_disc

    # ---------- 6.6 Step 2b: discriminator training ----------
    def train_discriminator(
        self, X_disc: np.ndarray, y_disc: np.ndarray
    ) -> float:
        """Train discriminator from scratch each round.

        STUB — depends on CNN forward/backward.
        """
        raise NotImplementedError(
            "train_discriminator deferred: requires real TrafficCNN."
        )

    # ---------- 6.7 Step 3: filter and predict ----------
    def filter_and_predict(self) -> np.ndarray:
        """Per open sample: discriminator decides familiar/unfamiliar; classifier
        assigns a hard label when familiar, -1 otherwise. Returns (N_open,).

        STUB — depends on CNN forward.
        """
        raise NotImplementedError(
            "filter_and_predict deferred: requires real TrafficCNN."
        )

    # ---------- 6.8 Step 5: distillation ----------
    def run_distillation(self, global_labels: np.ndarray) -> float:
        """Fine-tune classifier on open data using server's voted labels as target.

        Round-1 short-circuit (live, not stubbed): when all global labels are -1
        (no consensus yet), return 0.0 without constructing a DataLoader or
        stepping the optimizer.
        """
        valid_mask: np.ndarray = global_labels != -1
        if int(valid_mask.sum()) == 0:
            return 0.0
        raise NotImplementedError(
            "run_distillation CNN steps deferred: requires real TrafficCNN."
        )

    # ---------- 6.9 Flower channel: client -> server ----------
    def get_parameters(self, config: dict) -> List[np.ndarray]:
        """Repurposed Flower hook: return this client's hard-label predictions on
        the open data, wrapped in a single-element list.

        Note: this calls `filter_and_predict`, so it inherits its CNN dependency.
        """
        hard_labels: np.ndarray = self.filter_and_predict()
        return [hard_labels.astype(np.int64)]

    # ---------- 6.10 Flower channel: server -> client ----------
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Repurposed Flower hook: receive global voted labels from the server."""
        self.global_labels = parameters[0].astype(np.int64)

    # ---------- 6.11 fit: the 5-step orchestrator ----------
    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        """Execute all 5 SSFL steps for one round. Returns hard labels upstream."""
        self.current_round += 1
        logger.info(
            "[Client %d] round %d begin", self.client_id, self.current_round
        )

        # Receive voted labels from previous round.
        self.set_parameters(parameters)
        current_global_labels: np.ndarray = self.global_labels  # type: ignore[assignment]

        # Step 1: train classifier, compute confidence scores, derive threshold.
        clf_loss: float = self.train_classifier()
        confidence_scores: np.ndarray = self.compute_confidence_scores()
        threshold: float = self.compute_confidence_threshold(confidence_scores)

        # Step 2: build familiar/unfamiliar training set, train discriminator.
        X_disc, y_disc = self.build_discriminator_dataset(
            confidence_scores, threshold
        )
        disc_loss: float = self.train_discriminator(X_disc, y_disc)

        # Step 3: filter + predict hard labels for open samples.
        hard_labels: np.ndarray = self.filter_and_predict()

        # Step 5: distillation on globally-labelled open samples from last round.
        distill_loss: float = self.run_distillation(current_global_labels)

        n_familiar: int = int((hard_labels != -1).sum())
        n_unfamiliar: int = int(self.N_open - n_familiar)

        metrics: dict = {
            "classifier_loss": float(clf_loss),
            "discriminator_loss": float(disc_loss),
            "distillation_loss": float(distill_loss),
            "n_familiar": n_familiar,
            "n_unfamiliar": n_unfamiliar,
            "client_id": int(self.client_id),
        }
        return [hard_labels.astype(np.int64)], int(len(self.X_private)), metrics

    # ---------- 6.12 evaluate ----------
    def evaluate(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[float, int, dict]:
        """Local validation on the last 20% of private data.

        STUB — depends on CNN forward.
        """
        raise NotImplementedError(
            "evaluate deferred: requires real TrafficCNN."
        )
