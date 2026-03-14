"""
model.py  --  Shared model definition (copy this file to all PCs)
------------------------------------------------------------------
EdgeFed split architecture (Ye et al., 2020):

  low_features  : features[0] of MobileNetV2   -- run on CLIENT (CPU only)
                  Input  (B, 3, 224, 224) -> Output (B, 32, 112, 112)
                  One tiny ConvBNActivation block. Negligible compute.

  high_features : features[1:] of MobileNetV2  -- run on SERVER
  avgpool + classifier                          -- run on SERVER
                  Input  (B, 32, 112, 112) -> Output (B, NUM_CLASSES)
                  Full backward pass + SGD happens here.

PlantNet.forward_low(x)  -- client side only (no gradient needed)
PlantNet.forward_high(x) -- server side only (full backward pass)
PlantNet.forward(x)      -- full pipeline for inference / evaluation
"""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 38  # PlantVillage: 38 plant/disease categories


class PlantNet(nn.Module):
    """
    MobileNetV2 split at features[0] for EdgeFed computation offloading.

    pretrained=True  -> start from ImageNet weights (recommended).
    pretrained=False -> random init (e.g. when loading a saved checkpoint).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # ---- split point ----
        # Low  : features[0]   (first ConvBNActivation: 3->32 ch, stride 2)
        # High : features[1:]  (17 InvertedResidual blocks)
        self.low_features  = nn.Sequential(backbone.features[0])
        self.high_features = nn.Sequential(*list(backbone.features.children())[1:])
        self.avgpool       = nn.AdaptiveAvgPool2d((1, 1))

        # Replace the classification head for num_classes outputs
        in_features = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward_low(self, x):
        """CLIENT side -- run only the first conv block. No gradient needed."""
        return self.low_features(x)

    def forward_high(self, x):
        """SERVER side -- run high layers on aggregated activations x_conv."""
        x = self.high_features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        """Full forward pass -- for inference and evaluation."""
        return self.forward_high(self.forward_low(x))
