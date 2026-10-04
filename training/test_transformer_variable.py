import torch

from models.temporal_transformer import (
    TemporalTransformer
)


model = TemporalTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    num_classes=39
)


# Simulate a padded batch
x = torch.randn(
    3,
    31,
    960
)


# False = real
# True = padding

padding_mask = torch.tensor([
    [False] * 24 + [True] * 7,
    [False] * 31,
    [False] * 18 + [True] * 13
])


output = model(
    x,
    padding_mask=padding_mask
)


print("Input shape:")
print(x.shape)

print()

print("Mask shape:")
print(padding_mask.shape)

print()

print("Output shape:")
print(output.shape)