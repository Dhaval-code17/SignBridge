from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.cnn_baseline import CNNBaseline
from training.dataset import SignVideoDataset


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"
CLASS_FILE = PROJECT_ROOT / "configs" / "classes.json"

MODEL_DIR = PROJECT_ROOT / "models" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "cnn_baseline_best.pth"

SEED = 42
NUM_FRAMES = 16
NUM_CLASSES = 39

BATCH_SIZE = 4
NUM_WORKERS = 0

EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred for our baseline experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_mapping() -> dict[str, int]:
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_to_idx = data["class_to_idx"]

    if len(class_to_idx) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(class_to_idx)}."
        )

    return class_to_idx


# ============================================================
# MODEL FOR VIDEO
# ============================================================

def video_forward(
    model: nn.Module,
    frames: torch.Tensor,
) -> torch.Tensor:
    """
    Process all frames using the CNN and average frame logits.

    Input:
        frames = [B, T, C, H, W]

    Output:
        logits = [B, num_classes]
    """

    batch_size, time_steps, channels, height, width = frames.shape

    # Merge batch and temporal dimensions.
    frames = frames.view(
        batch_size * time_steps,
        channels,
        height,
        width,
    )

    frame_logits = model(frames)

    # Restore temporal dimension.
    frame_logits = frame_logits.view(
        batch_size,
        time_steps,
        -1,
    )

    # Temporal average.
    video_logits = frame_logits.mean(dim=1)

    return video_logits


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[float, float, float]:

    model.train()

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for frames, labels in loader:

        frames = frames.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision on NVIDIA GPU.
        with torch.autocast(
            device_type="cuda",
            enabled=DEVICE.type == "cuda",
        ):
            logits = video_forward(
                model,
                frames,
            )

            loss = criterion(
                logits,
                labels,
            )

        if DEVICE.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            labels.detach().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )

    epoch_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, accuracy, macro_f1


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:

    model.eval()

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for frames, labels in loader:

        frames = frames.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        with autocast(
            device_type="cuda",
            enabled=DEVICE.type == "cuda",
        ):
            logits = video_forward(
                model,
                frames,
            )

            loss = criterion(
                logits,
                labels,
            )

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    epoch_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, accuracy, macro_f1


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed(SEED)

    print("=" * 60)
    print("CNN BASELINE TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    class_to_idx = load_class_mapping()

    train_dataset = SignVideoDataset(
        csv_path=TRAIN_CSV,
        class_to_idx=class_to_idx,
        num_frames=NUM_FRAMES,
        training=True,
    )

    val_dataset = SignVideoDataset(
        csv_path=VAL_CSV,
        class_to_idx=class_to_idx,
        num_frames=NUM_FRAMES,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    model = CNNBaseline(
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = GradScaler(
    "cuda",
    enabled=DEVICE.type == "cuda",
)

    best_val_f1 = -1.0

    print("\nTrain samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Classes:", NUM_CLASSES)
    print("Batch size:", BATCH_SIZE)
    print("Frames/video:", NUM_FRAMES)
    print("Epochs:", EPOCHS)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_acc:.4f} | "
            f"Macro-F1: {train_f1:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"Accuracy: {val_acc:.4f} | "
            f"Macro-F1: {val_f1:.4f}"
        )

        # Save the best model using validation F1.
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "class_to_idx": class_to_idx,
                "num_frames": NUM_FRAMES,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                f"Saved best model → "
                f"{BEST_MODEL_PATH}"
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        "Best checkpoint:",
        BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()