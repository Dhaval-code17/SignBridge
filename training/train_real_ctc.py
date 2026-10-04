import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc

from models.ctc_transformer import CTCTransformer


# --------------------------------------------------
# Settings
# --------------------------------------------------

TRAIN_DIR = "data/continuous/train"
VAL_DIR = "data/continuous/val"

CLASSES_PATH = "data/person1/classes.json"

CHECKPOINT_DIR = "models/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "real_ctc_transformer.pth"
)

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load classes
# --------------------------------------------------

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

NUM_CLASSES = len(class_to_idx)

# CTC blank is after the 39 sign classes
BLANK_ID = NUM_CLASSES


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = RealCTCDataset(
    data_dir=TRAIN_DIR,
    class_to_idx=class_to_idx
)

val_dataset = RealCTCDataset(
    data_dir=VAL_DIR,
    class_to_idx=class_to_idx
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

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


# --------------------------------------------------
# Model
# --------------------------------------------------

model = CTCTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=NUM_CLASSES,
    dropout=0.1
)

model = model.to(DEVICE)


# --------------------------------------------------
# CTC loss
# --------------------------------------------------

criterion = nn.CTCLoss(
    blank=BLANK_ID,
    zero_infinity=True
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

print("======================================")
print("Real CTC Training")
print("======================================")

print("Device:", DEVICE)
print("Training sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))
print("Number of classes:", NUM_CLASSES)
print("CTC blank ID:", BLANK_ID)


for epoch in range(NUM_EPOCHS):

    # ------------------------------
    # Training
    # ------------------------------

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
        feature_lengths = feature_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)

        # CTC expects:
        # [time, batch, classes]

        log_probs = torch.log_softmax(
            logits,
            dim=-1
        )

        log_probs = log_probs.permute(
            1, 0, 2
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


    # ------------------------------
    # Validation
    # ------------------------------

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
            feature_lengths = feature_lengths.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            logits = model(features)

            log_probs = torch.log_softmax(
                logits,
                dim=-1
            )

            log_probs = log_probs.permute(
                1, 0, 2
            )

            loss = criterion(
                log_probs,
                targets,
                feature_lengths,
                target_lengths
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"- Train Loss: {train_loss:.4f} "
        f"- Val Loss: {val_loss:.4f}"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    CHECKPOINT_PATH
)

print()
print("Training complete!")
print("Model saved to:")
print(CHECKPOINT_PATH)