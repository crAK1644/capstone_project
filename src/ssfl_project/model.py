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
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Filtre sayılarını 64'ten 32'ye çekiyoruz
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=23, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        # Giriş: 32 kanal * 5 zaman adımı = 160
        self.fc1 = nn.Linear(in_features=160, out_features=64) # Nöron sayısını da 128'den 64'e düşürdük
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc1_relu(out)
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