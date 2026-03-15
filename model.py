"""
model.py  --  Shared model definition (copy this file to all PCs)
-----------------------------------------------------------------
Async Federated Learning architecture:

  Server  : PlantNet (MobileNetV2), trains on its own data continuously.
  Clients : PlantNet with optional backbone freeze for fast CPU training.
            Only the last N InvertedResidual blocks + classifier are trained,
            reducing trainable params from ~3.4M to ~0.3M (~10x speedup).

  Both use the SAME architecture -> direct FedAvg on state_dict works.

NOTE: State-dict keys changed from EdgeFed (low_features/high_features)
      to unified "features" -- old EdgeFed checkpoints are incompatible.
"""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 38  # PlantVillage: 38 plant/disease categories


class PlantNet(nn.Module):
    """
    MobileNetV2 for PlantVillage classification.

    pretrained=True  -> ImageNet weights (recommended for server).
    pretrained=False -> random init (used when loading a saved checkpoint).

    For CPU-only clients call freeze_for_client() after loading weights.
    It freezes the heavy backbone and only keeps the last few blocks +
    classifier trainable, giving ~10x speedup with minimal accuracy loss.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights  = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        self.features   = backbone.features                   # 19-block Sequential
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        in_features     = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    # ---------------------------------------------------------------- forward

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    # ---------------------------------------------------------------- client helpers

    def freeze_for_client(self, train_last_n_blocks: int = 3):
        """
        Freeze all except the last N feature blocks + classifier.

        MobileNetV2 features has 19 children (indices 0-18).
        Default: freeze features[0:16], train features[16:18] + classifier.

        Trainable param count drops from ~3.4M to ~0.3M -- very fast on CPU.
        """
        for p in self.parameters():
            p.requires_grad = False
        children = list(self.features.children())
        for block in children[-train_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters (default state for server training)."""
        for p in self.parameters():
            p.requires_grad = True

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
