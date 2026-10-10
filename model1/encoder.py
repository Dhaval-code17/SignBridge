"""
Model 1: Visual Feature Extractor for SignBridge.
Backbone: MobileNetV3-Large
Output: Frame-level visual feature embeddings [B, 960]
"""

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGE_SIZE = 224

# Default MobileNetV3 Normalization
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class Model1Encoder(nn.Module):
    """
    SignBridge Model 1 Encoder.
    Uses MobileNetV3-Large feature extractor.
    Returns:
        [B, 960] float32 feature tensor.
    """
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
