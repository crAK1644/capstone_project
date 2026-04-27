"""CNN architecture — implementation pass 2.

8-Conv + 2-FC TrafficCNN as described in section 5.1 of
docs/SSFL_FLOWER_INFRASTRUCTURE_PLAN.md. Used as both:
  - classifier    (num_classes=11) — predicts traffic class
  - discriminator (num_classes=2)  — familiar (0) vs unfamiliar (1)
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn


class TrafficCNN(nn.Module):
    """CNN used as classifier (num_classes=11) or discriminator (num_classes=2).

    Input  shape : (batch, 23, 5)        # 23 channels, length-5 sequence
    Output shape : (batch, num_classes)  # raw logits (no softmax)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes: int = num_classes

        # ----- Blocks 1-4: 64 channels (padding=1 keeps length=5) -----
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=23, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # ----- Block 5 upgrades width 64 -> 128 -----
        self.conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # ----- Blocks 6-8: 128 channels -----
        self.conv_block6 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv_block7 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv_block8 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # ----- Head: (batch,128,5) -> (batch,640) -> 128 -> num_classes -----
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=640, out_features=128)
        self.fc1_relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.conv_block6(out)
        out = self.conv_block7(out)
        out = self.conv_block8(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def build_classifier(num_classes: int, device: torch.device) -> TrafficCNN:
    """Factory for the classifier (output size = num_classes, e.g. 11)."""
    model = TrafficCNN(num_classes=num_classes).to(device)
    return model


def build_discriminator(device: torch.device) -> TrafficCNN:
    """Factory for the discriminator (output size = 2: familiar / unfamiliar)."""
    model = TrafficCNN(num_classes=2).to(device)
    return model


# ---------- Local checkpoint helpers (NOT the Flower wire) ----------
def get_model_parameters(model: TrafficCNN) -> List[np.ndarray]:
    """Snapshot full model state — parameters AND buffers — as numpy arrays."""
    state = model.state_dict()
    return [t.detach().cpu().numpy() for t in state.values()]


def set_model_parameters(
    model: TrafficCNN, parameters: List[np.ndarray]
) -> None:
    """Restore model state from a list previously produced by get_model_parameters."""
    state_keys: List[str] = list(model.state_dict().keys())
    if len(state_keys) != len(parameters):
        raise ValueError(
            f"set_model_parameters: expected {len(state_keys)} arrays, "
            f"got {len(parameters)}"
        )
    new_state = {
        k: torch.as_tensor(v) for k, v in zip(state_keys, parameters)
    }
    model.load_state_dict(new_state, strict=True)