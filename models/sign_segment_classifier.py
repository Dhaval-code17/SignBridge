import torch
import torch.nn as nn


class SignSegmentClassifier(nn.Module):
    def __init__(
        self,
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=39,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(self, x, lengths):

        output, _ = self.lstm(x)

        # Get the actual final frame for each sequence.
        last_indices = lengths - 1

        batch_indices = torch.arange(
            x.size(0),
            device=x.device,
        )

        last_output = output[
            batch_indices,
            last_indices,
        ]

        logits = self.classifier(last_output)

        return logits