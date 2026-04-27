"""Flower NumPyClient — SSFLClient orchestrates all 5 SSFL steps per round.

The orchestration (`fit`) and Flower-channel plumbing (`get_parameters`,
`set_parameters`) are fully wired. Per-step methods that require a real
CNN forward/backward (`train_classifier`, `compute_confidence_scores`,
`train_discriminator`, `filter_and_predict`, `run_distillation` beyond
the round-1 short-circuit, and `evaluate`) raise NotImplementedError until
model.py ships.
"""
import logging
import time
from typing import List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from config import DISTILLATION_EPOCHS
from metrics import payload_bytes_packed, payload_bytes_wire
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

        Returns the average cross-entropy loss over the final epoch.
        """
        # 1) Build a DataLoader over (X_private, y_private). Shuffle each epoch.
        X_tensor = torch.as_tensor(self.X_private, dtype=torch.float32)
        y_tensor = torch.as_tensor(self.y_private, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        private_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # 2) Loss function. CrossEntropyLoss expects raw logits (no softmax).
        criterion = torch.nn.CrossEntropyLoss()

        # 3) Put model in training mode (enables BatchNorm running-stat updates,
        #    enables Dropout).
        self.classifier.train()

        avg_loss: float = 0.0
        # 4) Epoch loop.
        for epoch in range(self.classifier_epochs):
            total_loss: float = 0.0
            n_batches: int = 0

            # 5) Batch loop.
            for X_batch, y_batch in private_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass.
                logits = self.classifier(X_batch)            # (B, num_classes)
                loss = criterion(logits, y_batch)            # scalar tensor

                # Backward pass + optimizer step.
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            # Average loss over this epoch (guard against empty loader).
            avg_loss = total_loss / max(n_batches, 1)

        # Return the FINAL epoch's average loss (per plan §6.2).
        return float(avg_loss)

    # ---------- 6.3 Step 1b: confidence scores ----------
    def compute_confidence_scores(self) -> np.ndarray:
        """Max-softmax confidence per open sample; shape (N_open,).

        Confidence = max softmax probability across classes. High = the
        classifier is sure about this sample; low = unfamiliar.
        """
        # 1) Build a DataLoader over X_open. shuffle=False is critical:
        #    the returned scores are indexed by position in X_open, so
        #    order must be preserved.
        X_tensor = torch.as_tensor(self.X_open, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        open_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        # 2) Inference mode: disables Dropout and freezes BatchNorm running stats.
        self.classifier.eval()

        confidence_scores: List[float] = []

        # 3) torch.no_grad() disables autograd → faster, less memory.
        with torch.no_grad():
            for (X_batch,) in open_loader:
                X_batch = X_batch.to(self.device)

                logits = self.classifier(X_batch)              # (B, num_classes)
                probs = torch.softmax(logits, dim=1)           # (B, num_classes)
                max_probs = probs.max(dim=1).values            # (B,)

                confidence_scores.extend(max_probs.cpu().tolist())

        return np.asarray(confidence_scores, dtype=np.float32)

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

        X_disc/y_disc come from build_discriminator_dataset (familiar=0,
        unfamiliar=1). Adam is rebuilt each call so previous-round momentum
        does not bleed in — the definition of 'familiar' shifts as the
        classifier improves.

        Returns the average binary cross-entropy loss over the final epoch.
        """
        # 1) Build a DataLoader over (X_disc, y_disc).
        X_tensor = torch.as_tensor(X_disc, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_disc, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        disc_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # 2) Loss + fresh Adam per round (per plan §6.1).
        criterion = torch.nn.CrossEntropyLoss()
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )

        # 3) Training mode for BatchNorm / Dropout.
        self.discriminator.train()

        avg_loss: float = 0.0
        # 4) Epoch loop.
        for epoch in range(self.discriminator_epochs):
            total_loss: float = 0.0
            n_batches: int = 0

            # 5) Batch loop.
            for X_batch, y_batch in disc_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.discriminator(X_batch)         # (B, 2)
                loss = criterion(logits, y_batch)            # scalar

                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

        return float(avg_loss)

    # ---------- 6.7 Step 3: filter and predict ----------
    def filter_and_predict(self) -> np.ndarray:
        """Per open sample: discriminator decides familiar/unfamiliar; classifier
        assigns a hard label when familiar, -1 otherwise. Returns (N_open,).

        Output convention (matches strategy.vote_mechanism):
            label in {0, 1, ..., num_classes-1}  -> familiar, this client votes
            label == -1                          -> unfamiliar, abstain
        """
        # 1) DataLoader, shuffle=False so output positions match X_open[i].
        X_tensor = torch.as_tensor(self.X_open, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        open_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        # 2) Both networks in eval mode (no dropout, frozen BN stats).
        self.classifier.eval()
        self.discriminator.eval()

        familiar_mask_chunks: List[np.ndarray] = []
        class_pred_chunks: List[np.ndarray] = []

        # 3) Two forward passes per batch, no autograd.
        with torch.no_grad():
            for (X_batch,) in open_loader:
                X_batch = X_batch.to(self.device)

                # Discriminator: 0 = familiar, 1 = unfamiliar.
                disc_logits = self.discriminator(X_batch)         # (B, 2)
                disc_pred = disc_logits.argmax(dim=1)             # (B,)
                familiar_mask = (disc_pred == 0)                  # (B,) bool

                # Classifier: hard label in [0, num_classes-1].
                clf_logits = self.classifier(X_batch)             # (B, num_classes)
                clf_pred = clf_logits.argmax(dim=1)               # (B,)

                familiar_mask_chunks.append(familiar_mask.cpu().numpy())
                class_pred_chunks.append(clf_pred.cpu().numpy())

        # 4) Concatenate, then mask: keep classifier label where familiar,
        #    write -1 elsewhere.
        familiar_mask_all = np.concatenate(familiar_mask_chunks, axis=0)  # (N_open,) bool
        class_pred_all = np.concatenate(class_pred_chunks, axis=0)        # (N_open,) int

        hard_labels = np.where(
            familiar_mask_all,
            class_pred_all,
            -1,
        ).astype(np.int64)

        return hard_labels

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

        # Keep only open samples that received a real vote.
        X_distill = self.X_open[valid_mask]
        y_distill = global_labels[valid_mask].astype(np.int64)

        X_tensor = torch.as_tensor(X_distill, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_distill, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        distill_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = torch.nn.CrossEntropyLoss()
        # Re-use the persistent classifier optimizer (momentum carries over).
        self.classifier.train()

        avg_loss: float = 0.0
        for epoch in range(self.distillation_epochs):
            total_loss: float = 0.0
            n_batches: int = 0
            for X_batch, y_batch in distill_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.classifier(X_batch)
                loss = criterion(logits, y_batch)

                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)

        return float(avg_loss)

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
        """Execute all 5 SSFL steps for one round. Returns hard labels upstream.

        Per-round instrumentation (see metrics.py):
            bytes_upload_wire   : actual Flower/gRPC upload cost (int64 * N_open)
            bytes_upload_packed : paper-fair upload cost (uint8 * N_open)
            confidence_threshold: this client's adaptive θ (median of scores)
            fit_wall_clock_sec  : end-to-end wall time spent in this method
        These fields are consumed by `SSFLStrategy.aggregate_fit` to build
        the `CommCostLedger` and per-round classification summary.
        """
        round_t0: float = time.perf_counter()
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

        # Payload accounting: what this single client contributes to Table IV.
        hard_labels_int64: np.ndarray = hard_labels.astype(np.int64)
        upload_bytes_wire: int = payload_bytes_wire(hard_labels_int64)
        upload_bytes_packed: int = payload_bytes_packed(hard_labels_int64)

        metrics: dict = {
            "classifier_loss": float(clf_loss),
            "discriminator_loss": float(disc_loss),
            "distillation_loss": float(distill_loss),
            "n_familiar": n_familiar,
            "n_unfamiliar": n_unfamiliar,
            "client_id": int(self.client_id),
            "bytes_upload_wire": int(upload_bytes_wire),
            "bytes_upload_packed": int(upload_bytes_packed),
            "confidence_threshold": float(threshold),
            "fit_wall_clock_sec": float(time.perf_counter() - round_t0),
        }
        return [hard_labels_int64], int(len(self.X_private)), metrics

    # ---------- 6.12 evaluate ----------
    def evaluate(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[float, int, dict]:
        """Local validation on the last 20% of private data.

        Returns (loss, num_examples, metrics_dict) per Flower's NumPyClient
        contract. `parameters` is ignored (Flower channel carries voted
        labels, not weights, and we already consumed them in fit()).
        """
        # 1) Last 20% of private data as a held-out validation slice.
        n_total: int = int(len(self.X_private))
        n_val: int = max(1, n_total // 5)
        X_val = self.X_private[-n_val:]
        y_val = self.y_private[-n_val:]

        X_tensor = torch.as_tensor(X_val, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_val, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.classifier.eval()

        total_loss: float = 0.0
        n_correct: int = 0
        n_seen: int = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.classifier(X_batch)
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=1)

                total_loss += float(loss.item())
                n_correct += int((preds == y_batch).sum().item())
                n_seen += int(y_batch.numel())

        avg_loss: float = total_loss / max(n_seen, 1)
        accuracy: float = n_correct / max(n_seen, 1)

        metrics: dict = {
            "accuracy": float(accuracy),
            "client_id": int(self.client_id),
        }
        return float(avg_loss), int(n_seen), metrics
