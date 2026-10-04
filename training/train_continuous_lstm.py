import sys
import os
import json

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc


# =========================
# Configuration
# =========================

TRAIN_DIR = "data/continuous/train"
VAL_DIR = "data/continuous/val"

CLASSES_PATH = "data/person1/classes.json"

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001

INPUT_DIM = 960
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 39
BLANK_ID = 39

CHECKPOINT_PATH = "models/checkpoints/continuous_lstm.pth"


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# Load classes
# =========================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_config = json.load(f)

class_to_idx = class_config["class_to_idx"]


# =========================
# Dataset
# =========================

train_dataset = RealCTCDataset(
    TRAIN_DIR,
    class_to_idx
)

val_dataset = RealCTCDataset(
    VAL_DIR,
    class_to_idx
)


# =========================
# DataLoaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_real_ctc
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_real_ctc
)


# =========================
# Continuous LSTM
# =========================

class ContinuousLSTM(nn.Module):

    def __init__(
        self,
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=39
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes + 1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        logits = self.classifier(output)

        return logits


model = ContinuousLSTM(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
).to(device)


# =========================
# Loss and optimizer
# =========================

criterion = nn.CTCLoss(
    blank=BLANK_ID,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training
# =========================

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for (
        features,
        feature_lengths,
        targets,
        target_lengths
    ) in train_loader:

        features = features.to(device)
        feature_lengths = feature_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        logits = model(features)

        # CTC expects [T, B, C]
        logits = logits.transpose(0, 1)

        log_probs = torch.log_softmax(
            logits,
            dim=2
        )

        loss = criterion(
            log_probs,
            targets,
            feature_lengths,
            target_lengths
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # =========================
    # Validation
    # =========================

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for (
            features,
            feature_lengths,
            targets,
            target_lengths
        ) in val_loader:

            features = features.to(device)
            feature_lengths = feature_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(features)

            logits = logits.transpose(0, 1)

            log_probs = torch.log_softmax(
                logits,
                dim=2
            )

            loss = criterion(
                log_probs,
                targets,
                feature_lengths,
                target_lengths
            )

            total_val_loss += loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Train Loss: {average_train_loss:.4f} "
        f"- Val Loss: {average_val_loss:.4f}"
    )


# =========================
# Save model
# =========================

os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True
)

torch.save(
    model.state_dict(),
    CHECKPOINT_PATH
)

print(
    f"Saved model to {CHECKPOINT_PATH}"
)