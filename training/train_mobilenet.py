from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.mobilenet_model import MobileNetV3SignRecognizer
from training.dataset import SignVideoDataset


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"
CLASS_FILE = PROJECT_ROOT / "configs" / "classes.json"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "mobilenet_v3_stage1_best.pth"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

NUM_CLASSES = 39
NUM_FRAMES = 16

BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CLASS MAPPING
# ============================================================

def load_class_mapping() -> dict[str, int]:
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = data["class_to_idx"]

    if len(mapping) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(mapping)}."
        )

    return mapping


# ============================================================
# VIDEO FORWARD
# ============================================================

def video_forward(
    model: MobileNetV3SignRecognizer,
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process all frames in a video.

    Input:
        [B, T, C, H, W]

    Returns:
        video_logits:
            [B, num_classes]

        video_features:
            [B, T, feature_dim]
    """

    batch_size, time_steps, channels, height, width = (
        frames.shape
    )

    flat_frames = frames.reshape(
        batch_size * time_steps,
        channels,
        height,
        width,
    )

    frame_logits, frame_features = model(
        flat_frames
    )

    frame_logits = frame_logits.reshape(
        batch_size,
        time_steps,
        -1,
    )

    frame_features = frame_features.reshape(
        batch_size,
        time_steps,
        -1,
    )

    # Average frame logits for the isolated-sign classifier.
    video_logits = frame_logits.mean(dim=1)

    return video_logits, frame_features


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model: MobileNetV3SignRecognizer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[float, float, float]:

    model.train()

    total_loss = 0.0

    all_targets: list[int] = []
    all_predictions: list[int] = []

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

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
        ):
            logits, _ = video_forward(
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

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
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
# VALIDATE
# ============================================================

@torch.no_grad()
def validate(
    model: MobileNetV3SignRecognizer,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:

    model.eval()

    total_loss = 0.0

    all_targets: list[int] = []
    all_predictions: list[int] = []

    for frames, labels in loader:

        frames = frames.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda",
        ):
            logits, _ = video_forward(
                model,
                frames,
            )

            loss = criterion(
                logits,
                labels,
            )

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
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
    print("MOBILENETV3 STAGE 1 TRAINING")
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

    model = MobileNetV3SignRecognizer(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
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
    print("Frames/video:", NUM_FRAMES)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc, train_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
            )
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
                "feature_dim": model.feature_dim,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                "Saved best model →",
                BEST_MODEL_PATH,
            )

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)

    print(
        f"Best validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        "Checkpoint:",
        BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()