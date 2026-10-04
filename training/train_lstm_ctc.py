import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc
from training.temporal_augmentation import temporal_augment
from models.ctc_lstm import CTCLSTM


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
    "models/checkpoints/"
    "new_lstm_ctc.pth"
)


# ============================================================
# Training settings
# ============================================================

EPOCHS = 20

BATCH_SIZE = 4

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.01

GRAD_CLIP = 5.0


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load classes
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:

    classes_data = json.load(f)


class_to_idx = classes_data["class_to_idx"]

num_classes = classes_data["num_classes"]

blank_id = num_classes


# ============================================================
# Dataset
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

model = CTCLSTM(
    input_dim=960,
    hidden_dim=256,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.3
)

model.to(device)


# ============================================================
# CTC loss
# ============================================================

criterion = nn.CTCLoss(
    blank=blank_id,
    zero_infinity=True
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")


print("======================================")
print("LSTM + CTC Training")
print("======================================")

print(f"Device: {device}")
print(f"Training sequences: {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Number of classes: {num_classes}")
print(f"CTC blank ID: {blank_id}")
print(f"Epochs: {EPOCHS}")
print("Temporal augmentation: ENABLED")
print()


for epoch in range(EPOCHS):

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

        features = features.to(device)

        # ----------------------------------------------------
        # Temporal augmentation
        # ----------------------------------------------------

        augmented_features = []

        for i in range(features.size(0)):

            sequence = features[
                i,
                :feature_lengths[i]
            ]

            sequence = temporal_augment(
                sequence
            )

            augmented_features.append(
                sequence
            )

        features = torch.nn.utils.rnn.pad_sequence(
            augmented_features,
            batch_first=True,
            padding_value=0.0
        )

        feature_lengths = torch.tensor(
            [
                x.size(0)
                for x in augmented_features
            ],
            dtype=torch.long,
            device=device
        )

        targets = targets.to(device)

        target_lengths = target_lengths.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        optimizer.zero_grad()

        logits = model(features)

        log_probs = torch.log_softmax(
            logits,
            dim=-1
        )

        log_probs = log_probs.transpose(
            0,
            1
        )

        # ----------------------------------------------------
        # CTC loss
        # ----------------------------------------------------

        loss = criterion(
            log_probs,
            targets,
            feature_lengths,
            target_lengths
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
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

            features = features.to(device)

            feature_lengths = (
                feature_lengths.to(device)
            )

            targets = targets.to(device)

            target_lengths = (
                target_lengths.to(device)
            )

            logits = model(features)

            log_probs = torch.log_softmax(
                logits,
                dim=-1
            )

            log_probs = log_probs.transpose(
                0,
                1
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
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Train Loss: {train_loss:.4f} "
        f"- Val Loss: {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # Save best checkpoint
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
            f"  Best model saved "
            f"(Val Loss: {val_loss:.4f})"
        )


print()
print("======================================")
print("Training complete!")
print("======================================")

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    "Best model saved to:"
)

print(CHECKPOINT_PATH)