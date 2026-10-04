import torch
import torch.nn as nn


class CTCLSTM(nn.Module):
    def __init__(
        self,
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=4764,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(
            hidden_dim * 2
        )

        self.classifier = nn.Linear(
            hidden_dim * 2,
            num_classes + 1
        )

    def forward(self, x):
        x, _ = self.lstm(x)

        x = self.layer_norm(x)

        logits = self.classifier(x)

        return logits