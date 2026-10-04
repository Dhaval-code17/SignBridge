from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


class MobileNetV3SignRecognizer(nn.Module):
    """
    MobileNetV3-Large based visual sign recognizer.

    Input:
        [B, 3, 224, 224]

    Outputs:
        logits:
            [B, num_classes]

        features:
            [B, feature_dim]

    The feature representation is kept available for Member 2.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be greater than zero."
            )

        weights = (
            MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = mobilenet_v3_large(
            weights=weights,
        )

        # MobileNetV3's final classifier begins with:
        # Linear(960 -> 1280)
        #
        # The input to that classifier is our 960-D embedding.
        self.backbone = backbone.features
        self.pool = backbone.avgpool

        self.feature_dim = 960

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(
                self.feature_dim,
                256,
            ),
            nn.Hardswish(),
            nn.Dropout(p=0.30),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract one 960-dimensional visual feature vector
        for each frame.

        Input:
            [B, 3, 224, 224]

        Output:
            [B, 960]
        """
        if x.ndim != 4:
            raise ValueError(
                "Expected input with shape "
                "[B, 3, H, W]. "
                f"Received {tuple(x.shape)}."
            )

        x = self.backbone(x)
        x = self.pool(x)

        # [B, 960, 1, 1] -> [B, 960]
        x = torch.flatten(x, 1)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits, features
        """
        features = self.extract_features(x)
        logits = self.classifier(features)

        return logits, features