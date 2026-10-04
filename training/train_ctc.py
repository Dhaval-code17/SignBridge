import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.ctc_dataset import CTCSyntheticDataset
from training.ctc_collate import collate_ctc
from models.ctc_transformer import CTCTransformer


# ------------------------------------------------
# Settings
# ------------------------------------------------

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 39
INPUT_DIM = 960

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ------------------------------------------------
# Dataset
# ------------------------------------------------

dataset = CTCSyntheticDataset(
    num_samples=200,
    min_length=20,
    max_length=40,
    feature_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    max_target_length=5
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_ctc
)


# ------------------------------------------------
# Model
# ------------------------------------------------

model = CTCTransformer(
    input_dim=INPUT_DIM,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=NUM_CLASSES,
    dropout=0.1
)

model = model.to(DEVICE)


# ------------------------------------------------
# CTC Loss
# ------------------------------------------------

criterion = nn.CTCLoss(
    blank=NUM_CLASSES,
    zero_infinity=True
)


# ------------------------------------------------
# Optimizer
# ------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------------------------------
# Training
# ------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for (
        features,
        feature_lengths,
        targets,
        target_lengths
    ) in loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        logits = model(features)

        # CTC expects:
        #
        # [time, batch, classes]
        #
        log_probs = torch.log_softmax(
            logits,
            dim=-1
        )

        log_probs = log_probs.transpose(
            0,
            1
        )

        # Calculate CTC loss
        loss = criterion(
            log_probs,
            targets,
            feature_lengths,
            target_lengths
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


# ------------------------------------------------
# Save model
# ------------------------------------------------

import os

os.makedirs(
    "models/checkpoints",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/checkpoints/ctc_transformer.pth"
)

print("\nModel saved to:")
print("models/checkpoints/ctc_transformer.pth")