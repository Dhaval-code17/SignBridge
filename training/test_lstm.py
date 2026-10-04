import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch

from models.lstm import LSTMClassifier


# Create model
model = LSTMClassifier(
    input_dim=960,
    hidden_dim=256,
    num_layers=2,
    num_classes=39
)

# Fake batch
x = torch.randn(
    32,
    16,
    960
)

# Forward pass
output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)