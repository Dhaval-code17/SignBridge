import torch

from models.temporal_transformer import (
    TemporalTransformer
)


# Create model
model = TemporalTransformer(
    input_dim=960,
    d_model=256,
    num_heads=8,
    num_layers=4,
    num_classes=39
)


# Test with Person 1's actual sequence length
x = torch.randn(
    2,
    24,
    960
)


output = model(x)


print("Input shape :", x.shape)
print("Output shape:", output.shape)