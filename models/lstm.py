import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=39,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # [batch, 16, 960]

        output, (hidden, cell) = self.lstm(x)

        # Take the final time step
        last_output = output[:, -1, :]

        # Classification
        logits = self.classifier(last_output)

        return logits