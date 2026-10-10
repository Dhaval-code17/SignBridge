"""
Model 2: Temporal Classifier Head & Attention Encoder for SignBridge.
Input: Visual feature sequence [B, T, 960]
Output: Class logits [B, num_classes] & Attention context vector [B, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceClassifierHead(nn.Module):
    """
    Temporal Classifier Head taking [B, T, 960] features
    and returning logits [B, num_classes] alongside 512-D attention context.
    """
    def __init__(self, in_dim=960, hidden_dim=512, num_classes=101):
        super().__init__()
        # Frame-level projection
        self.fc_frame = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        # Temporal Attention Weighting
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Output Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, 960]
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        h_flat = self.fc_frame(x_flat)  # [B*T, 512]
        h = h_flat.view(B, T, -1)      # [B, T, 512]

        # Temporal attention scores
        attn_logits = self.attn(h)                     # [B, T, 1]
        attn_weights = F.softmax(attn_logits, dim=1)   # [B, T, 1]

        # Weighted video representation
        context = torch.sum(h * attn_weights, dim=1)  # [B, 512]
        logits = self.classifier(context)             # [B, num_classes]
        return logits, context
