import torch

from models.ctc_transformer import CTCTransformer


model = CTCTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    num_classes=39
)


# Three sequences with different lengths
x = torch.randn(3, 31, 960)

padding_mask = torch.zeros(
    3,
    31,
    dtype=torch.bool
)

# Sequence 1
padding_mask[0, 24:] = True

# Sequence 2
padding_mask[1, :] = False

# Sequence 3
padding_mask[2, 18:] = True


logits = model(
    x,
    padding_mask=padding_mask
)


print("Input shape:")
print(x.shape)

print("\nMask shape:")
print(padding_mask.shape)

print("\nLogits shape:")
print(logits.shape)