import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape:
            [batch_size, sequence_length, d_model]
        """
        sequence_length = x.size(1)

        return x + self.pe[:, :sequence_length, :]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=960,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=39,
        dropout=0.1,
    ):
        super().__init__()

        # Person 1 output:
        # [B, T, 960]
        #
        # Convert 960-dimensional features to Transformer dimension.
        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # Positional encoding preserves temporal order.
        self.positional_encoding = PositionalEncoding(
            d_model=d_model
        )

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final classification layers.
        self.layer_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x, padding_mask=None):
        """
        Parameters
        ----------
        x:
            Tensor of shape [B, T, 960]

        padding_mask:
            Boolean tensor of shape [B, T].

            False = real feature
            True  = padding

        Returns
        -------
        logits:
            Tensor of shape [B, 39]
        """

        # [B, T, 960]
        # ->
        # [B, T, 256]
        x = self.input_projection(x)

        # Add temporal positional information.
        x = self.positional_encoding(x)

        # Transformer processes the temporal sequence.
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # Handle variable-length sequences.
        if padding_mask is not None:

            # True for valid/non-padding positions.
            valid_mask = (~padding_mask).unsqueeze(-1)

            # Remove padded positions from the pooled representation.
            x = x.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0
            )

            # Number of real frames in each sequence.
            valid_count = valid_mask.sum(
                dim=1
            ).clamp(min=1)

            # Mean pooling over real frames only.
            x = x.sum(dim=1) / valid_count

        else:
            # If there is no padding mask,
            # simply average over all frames.
            x = x.mean(dim=1)

        # Normalize before classification.
        x = self.layer_norm(x)

        # [B, 256] -> [B, 39]
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    # Simple test.

    model = TemporalTransformer(
        input_dim=960,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=39,
        dropout=0.1,
    )

    # Example variable-length batch.
    x = torch.randn(3, 31, 960)

    padding_mask = torch.zeros(
        3,
        31,
        dtype=torch.bool
    )

    # Sequence 1: 24 real frames + 7 padding frames
    padding_mask[0, 24:] = True

    # Sequence 2: all 31 frames are real
    padding_mask[1, :] = False

    # Sequence 3: 18 real frames + 13 padding frames
    padding_mask[2, 18:] = True

    output = model(
        x,
        padding_mask=padding_mask
    )

    print("Input shape:", x.shape)
    print("Padding mask shape:", padding_mask.shape)
    print("Output shape:", output.shape)