"""
model.py  —  Shared model definition (copy this file to all PCs)
─────────────────────────────────────────────────────────────────
MobileNetV2 fine-tuned for PlantVillage plant disease classification.
  Input  : 224x224 RGB image
  Output : NUM_CLASSES logits  (default 38 — PlantVillage)
"""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 38  # PlantVillage: 38 plant/disease categories


class PlantNet(nn.Module):
    """
    MobileNetV2 backbone with a replaced classification head.
    pretrained=True  → start from ImageNet weights (recommended).
    pretrained=False → random init (train from scratch).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Replace the final classifier layer
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.model = backbone

    def forward(self, x):
        return self.model(x)
