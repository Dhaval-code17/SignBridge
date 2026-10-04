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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

STAGE1_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_stage1_best.pth"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STAGE2_CHECKPOINT = (
    CHECKPOINT_DIR
    / "mobilenet_v3_stage2_best.pth"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

NUM_CLASSES = 39
NUM_FRAMES = 16

BATCH_SIZE = 2
NUM_WORKERS = 0

# Smoke test first.
# We will change this to 15 after the smoke test succeeds.
EPOCHS = 15

# Fine-tuning learning rates.
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 2.5e-4

WEIGHT_DECAY = 1e-4

# Only the final N backbone blocks are unfrozen.
NUM_UNFROZEN_BLOCKS = 4

EARLY_STOPPING_PATIENCE = 4

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
    with open(
        CLASS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    mapping = data["class_to_idx"]

    if len(mapping) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(mapping)}."
        )

    return mapping


# ============================================================
# MODEL FREEZING / UNFREEZING
# ============================================================

def configure_fine_tuning(
    model: MobileNetV3SignRecognizer,
    num_unfrozen_blocks: int,
) -> list[int]:

    if num_unfrozen_blocks <= 0:
        raise ValueError(
            "num_unfrozen_blocks must be positive."
        )

    backbone_blocks = list(
        model.backbone.children()
    )

    if num_unfrozen_blocks > len(backbone_blocks):
        raise ValueError(
            f"Requested {num_unfrozen_blocks} blocks, "
            f"but backbone only has "
            f"{len(backbone_blocks)} blocks."
        )

    # Freeze the complete backbone first.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    # Unfreeze only the final N blocks.
    start_index = (
        len(backbone_blocks)
        - num_unfrozen_blocks
    )

    unfrozen_indices = list(
        range(
            start_index,
            len(backbone_blocks),
        )
    )

    for index in unfrozen_indices:
        for parameter in backbone_blocks[
            index
        ].parameters():
            parameter.requires_grad = True

    # The classifier is always trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return unfrozen_indices


# ============================================================
# VIDEO FORWARD
# ============================================================

def video_forward(
    model: MobileNetV3SignRecognizer,
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

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

    # Isolated-sign classification:
    # average predictions across sampled frames.
    video_logits = frame_logits.mean(
        dim=1
    )

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

        optimizer.zero_grad(
            set_to_none=True
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

        if DEVICE.type == "cuda":

            scaler.scale(
                loss
            ).backward()

            # Prevent unusually large fine-tuning updates.
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

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

    return (
        epoch_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# VALIDATION
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
            loss.item()
            * labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        all_targets.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

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

    return (
        epoch_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed(SEED)

    print("=" * 60)
    print("MOBILENETV3 STAGE 2 FINE-TUNING")
    print("=" * 60)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Verify checkpoint exists.
    # --------------------------------------------------------

    if not STAGE1_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Stage-1 checkpoint not found:\n"
            f"{STAGE1_CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Load class mapping.
    # --------------------------------------------------------

    class_to_idx = load_class_mapping()

    # --------------------------------------------------------
    # Dataset.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create architecture WITHOUT downloading pretrained
    # weights. The Stage-1 checkpoint is our starting point.
    # --------------------------------------------------------

    model = MobileNetV3SignRecognizer(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=True,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load Stage-1 checkpoint.
    # --------------------------------------------------------

    checkpoint = torch.load(
        STAGE1_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Stage-1 checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "\nLoaded Stage-1 checkpoint:"
    )
    print(
        "Epoch:",
        checkpoint.get("epoch"),
    )
    print(
        "Validation accuracy:",
        checkpoint.get("val_accuracy"),
    )
    print(
        "Validation Macro-F1:",
        checkpoint.get("val_f1"),
    )

    # --------------------------------------------------------
    # Fine-tuning configuration.
    # --------------------------------------------------------

    unfrozen_indices = configure_fine_tuning(
        model,
        NUM_UNFROZEN_BLOCKS,
    )

    print(
        "\nUnfrozen backbone block indices:",
        unfrozen_indices,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Total parameters:",
        total_parameters,
    )

    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    # --------------------------------------------------------
    # Loss / optimizer.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]

    classifier_parameters = list(
        model.classifier.parameters()
    )

    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR,
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
    )

    scaler = GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda",
    )

    # --------------------------------------------------------
    # Evaluate the loaded Stage-1 checkpoint BEFORE training.
    # --------------------------------------------------------

    initial_loss, initial_acc, initial_f1 = validate(
        model,
        val_loader,
        criterion,
    )

    print("\nStage-1 checkpoint re-validation:")
    print(
        f"Loss: {initial_loss:.4f} | "
        f"Accuracy: {initial_acc:.4f} | "
        f"Macro-F1: {initial_f1:.4f}"
    )

    # --------------------------------------------------------
    # Training.
    # --------------------------------------------------------

    best_val_f1 = -1.0
    epochs_without_improvement = 0

    print("\nTrain samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Classes:", NUM_CLASSES)
    print("Frames/video:", NUM_FRAMES)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Backbone LR:", BACKBONE_LR)
    print("Classifier LR:", CLASSIFIER_LR)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        (
            train_loss,
            train_acc,
            train_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        (
            val_loss,
            val_acc,
            val_f1,
        ) = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_f1)

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

        current_backbone_lr = optimizer.param_groups[0][
            "lr"
        ]

        current_classifier_lr = optimizer.param_groups[1][
            "lr"
        ]

        print(
            f"LR    | "
            f"Backbone: {current_backbone_lr:.2e} | "
            f"Classifier: {current_classifier_lr:.2e}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            epochs_without_improvement = 0

            checkpoint_out = {
                "stage": 2,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "class_to_idx": class_to_idx,
                "num_frames": NUM_FRAMES,
                "feature_dim": model.feature_dim,
                "unfrozen_backbone_blocks": unfrozen_indices,
                "backbone_lr": BACKBONE_LR,
                "classifier_lr": CLASSIFIER_LR,
            }

            torch.save(
                checkpoint_out,
                STAGE2_CHECKPOINT,
            )

            print(
                "Saved best Stage-2 model ->",
                STAGE2_CHECKPOINT,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)

    print(
        f"Best Stage-2 validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        "Checkpoint:",
        STAGE2_CHECKPOINT,
    )


if __name__ == "__main__":
    main()