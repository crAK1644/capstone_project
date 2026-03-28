"""
CNN-based models for SSFL intrusion detection.

Architecture follows Table I of the paper:
- 8 convolutional layers (first 4: 64 filters, last 4: 128 filters, kernel 3×3)
- 3 fully-connected layers for classification
- Classifier outputs 11 classes (traffic categories)
- Discriminator outputs 2 classes (familiar / unfamiliar)

Both share the same CNN backbone; only the final FC layer differs.
"""

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    Shared 8-layer CNN feature extractor.

    Input shape:  (batch, 1, 23, 5)   — 1-channel, 23 features × 5 time windows
    Output shape: (batch, 128 * 23 * 5) — flattened feature map

    Architecture:
        Conv block 1 (layers 1-4): 64 kernels, 3×3, padding=1, ReLU
        Conv block 2 (layers 5-8): 128 kernels, 3×3, padding=1, ReLU
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()

        # --- Block 1: 4 conv layers, 64 filters each ---
        block1_layers = []
        for i in range(4):
            c_in = in_channels if i == 0 else 64
            block1_layers.append(nn.Conv2d(c_in, 64, kernel_size=3, padding=1))
            block1_layers.append(nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(*block1_layers)

        # --- Block 2: 4 conv layers, 128 filters each ---
        block2_layers = []
        for i in range(4):
            c_in = 64 if i == 0 else 128
            block2_layers.append(nn.Conv2d(c_in, 128, kernel_size=3, padding=1))
            block2_layers.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*block2_layers)

        # Flattened feature size: 128 channels × 23 height × 5 width
        self.flat_features = 128 * 23 * 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, 23, 5).

        Returns:
            Flattened feature tensor of shape (batch, 128*23*5).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # flatten
        return x


class Classifier(nn.Module):
    """
    CNN classifier for traffic intrusion detection.

    Uses CNNBackbone + 3-layer MLP head → 11 traffic categories.
    This is model w^{k,c} in the paper.
    """

    def __init__(self, num_classes: int = 11) -> None:
        super().__init__()
        self.backbone = CNNBackbone(in_channels=1)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, 23, 5).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        features = self.backbone(x)
        return self.head(features)


class Discriminator(nn.Module):
    """
    CNN discriminator for familiar/unfamiliar sample detection.

    Uses CNNBackbone + 3-layer MLP head → 2 classes (familiar, unfamiliar).
    This is model w^{k,d} in the paper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = CNNBackbone(in_channels=1)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, 23, 5).

        Returns:
            Logits tensor of shape (batch, 2).
        """
        features = self.backbone(x)
        return self.head(features)
