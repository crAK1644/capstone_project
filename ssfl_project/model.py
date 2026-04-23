"""CNN architecture — DECLARED but BODIES STUBBED.

This module is the *undeclared entry point* for the starting-infrastructure
pass (see SSFL_FLOWER_INFRASTRUCTURE_PLAN.md section 3.1). The signatures
are final. The bodies raise NotImplementedError until the CNN is delivered
in implementation pass 2.

Do NOT replace these stubs with a placeholder model. The surrounding
infrastructure (client.py, strategy.py, server.py, main.py) calls these
factories exactly as it will in production — the stubs exist so wiring
can be validated without freezing the model definition.
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn


class TrafficCNN(nn.Module):
    """CNN used as classifier (num_classes=11) or discriminator (num_classes=2).

    Planned architecture (per section 5.1):
        Conv1d(23, 64,  k=3, p=1) -> BN -> ReLU         [block 1]
        Conv1d(64, 64,  k=3, p=1) -> BN -> ReLU         [block 2]
        Conv1d(64, 64,  k=3, p=1) -> BN -> ReLU         [block 3]
        Conv1d(64, 64,  k=3, p=1) -> BN -> ReLU         [block 4]
        Conv1d(64, 128, k=3, p=1) -> BN -> ReLU         [block 5]
        Conv1d(128,128, k=3, p=1) -> BN -> ReLU         [block 6]
        Conv1d(128,128, k=3, p=1) -> BN -> ReLU         [block 7]
        Conv1d(128,128, k=3, p=1) -> BN -> ReLU         [block 8]
        Flatten                           # -> (batch, 640)
        Linear(640, 128) -> ReLU -> Dropout(0.5)
        Linear(128, num_classes)

    Input  shape : (batch, 23, 5)   [23 in-channels, 5 sequence length]
    Output shape : (batch, num_classes)   [raw logits]
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        raise NotImplementedError(
            "TrafficCNN body deferred to implementation pass 2 "
            "(CNN is the undeclared entry point of the starting infrastructure)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TrafficCNN.forward body deferred to implementation pass 2."
        )


def build_classifier(num_classes: int, device: torch.device) -> TrafficCNN:
    """Factory for the classifier (output size = num_classes, e.g. 11).

    STUB: raises NotImplementedError. Implement in pass 2 as:
        model = TrafficCNN(num_classes=num_classes).to(device)
        return model
    """
    raise NotImplementedError(
        "build_classifier deferred until TrafficCNN body is implemented."
    )


def build_discriminator(device: torch.device) -> TrafficCNN:
    """Factory for the discriminator (output size = 2: familiar / unfamiliar).

    STUB: raises NotImplementedError. Implement in pass 2 as:
        model = TrafficCNN(num_classes=2).to(device)
        return model
    """
    raise NotImplementedError(
        "build_discriminator deferred until TrafficCNN body is implemented."
    )


# ---------- Local checkpoint helpers (NOT the Flower wire) ----------
def get_model_parameters(model: TrafficCNN) -> List[np.ndarray]:
    """Snapshot the full model state — parameters AND buffers — as numpy arrays.

    Uses `state_dict()` so BatchNorm running statistics are captured. Used
    only for local checkpointing; the Flower channel in SSFL carries hard
    labels, not model weights.
    """
    state = model.state_dict()
    return [t.detach().cpu().numpy() for t in state.values()]


def set_model_parameters(
    model: TrafficCNN, parameters: List[np.ndarray]
) -> None:
    """Restore model state from a list previously produced by `get_model_parameters`.

    Uses `load_state_dict(..., strict=True)` so any mismatch fails loudly.
    """
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
