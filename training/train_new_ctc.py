import os
import sys
import json

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc
from models.ctc_transformer import CTCTransformer


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = (
    r"C:\Users\nihav\Downloads\final_model1_features"
    r"\configs\classes_final.json"
)

CHECKPOINT_DIR = "models/checkpoints"
CHECKPOINT_PATH = (
    "models/checkpoints/new_real_ctc_transformer.pth"
)


# ============================================================
# Training settings
# ============================================================

BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load classes
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

NUM_CLASSES = len(class_to_idx)
BLANK_ID = NUM_CLASSES


# ============================================================
# Datasets
# ============================================================

train_dataset = RealCTCDataset(
    TRAIN_DIR,
    class_to_idx
)

val_dataset = RealCTCDataset(
    VAL_DIR,
    class_to_idx
)


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


# ============================================================
# Model
# ============================================================

model = CTCTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=NUM_CLASSES,
    dropout=0.1,
)

model = model.to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CTCLoss(
    blank=BLANK_ID,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training information
# ============================================================

print("======================================")
print("New Real CTC Training")
print("======================================")
print(f"Device: {DEVICE}")
print(f"Training sequences: {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"CTC blank ID: {BLANK_ID}")
print(f"Epochs: {NUM_EPOCHS}")
print()


# ============================================================
# Best checkpoint tracking
# ============================================================

best_val_loss = float("inf")


# ============================================================
# Training loop
# ============================================================

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for (
        features,
        feature_lengths,
        targets,
        target_lengths
    ) in train_loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)

        log_probs = logits.log_softmax(
            dim=-1
        )

        log_probs = log_probs.transpose(
            0, 1
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

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for (
            features,
            feature_lengths,
            targets,
            target_lengths
        ) in val_loader:

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)

            log_probs = logits.log_softmax(
                dim=-1
            )

            log_probs = log_probs.transpose(
                0, 1
            )

            loss = criterion(
                log_probs,
                targets,
                feature_lengths,
                target_lengths
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"- Train Loss: {train_loss:.4f} "
        f"- Val Loss: {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        os.makedirs(
            CHECKPOINT_DIR,
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print(
            f"  Best model saved! "
            f"Val Loss: {best_val_loss:.4f}"
        )


# ============================================================
# Finished
# ============================================================

print()
print("Training complete!")
print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)
print(
    "Best model saved to:"
)
print(CHECKPOINT_PATH)