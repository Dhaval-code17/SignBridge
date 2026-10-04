from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "processed" / "final_dev_train.csv"
VAL_CSV = ROOT / "data" / "processed" / "final_dev_val.csv"
CLASSES_JSON = ROOT / "configs" / "classes_final.json"

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "mobilenet_v3_full_vocab_best.pth"
)

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 2
NUM_WORKERS = 0

MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3

FRAME_CHUNK = 8

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 3e-4

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

print("=" * 70)
print("SIGNBRIDGE — FULL VOCABULARY MOBILENETV3 TRAINING")
print("=" * 70)
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(f"PyTorch: {torch.__version__}")
print()


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    CLASSES_JSON,
    "r",
    encoding="utf-8",
) as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

NUM_CLASSES = len(class_to_idx)

assert NUM_CLASSES == 4764, (
    f"Expected 4764 classes, got {NUM_CLASSES}"
)

idx_to_class = {
    int(k): v
    for k, v in class_data["idx_to_class"].items()
}


# ============================================================
# LOAD CSVs
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

required_columns = {
    "uid",
    "gloss",
    "video_path",
}

for name, df in [
    ("train", train_df),
    ("validation", val_df),
]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: {missing}"
        )


# ============================================================
# LABEL VALIDATION
# ============================================================

unknown_train = (
    set(train_df["gloss"].astype(str))
    - set(class_to_idx)
)

unknown_val = (
    set(val_df["gloss"].astype(str))
    - set(class_to_idx)
)

if unknown_train:
    raise ValueError(
        f"Unknown training classes: {sorted(unknown_train)[:10]}"
    )

if unknown_val:
    raise ValueError(
        f"Unknown validation classes: {sorted(unknown_val)[:10]}"
    )


# ============================================================
# VIDEO PATH RESOLUTION
# ============================================================

def resolve_video_path(value: str) -> Path:
    path = Path(str(value))

    if path.is_absolute():
        return path

    candidates = [
        ROOT / path,
        ROOT / "data" / "raw" / "CISLR" / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


# ============================================================
# VERIFY VIDEO FILES BEFORE TRAINING
# ============================================================

print("Checking training video paths...")

missing_train = []

for p in train_df["video_path"]:
    path = resolve_video_path(p)

    if not path.exists():
        missing_train.append(str(path))

print(
    f"Training files checked: {len(train_df)}"
)
print(
    f"Missing training files: {len(missing_train)}"
)

if missing_train:
    print("\nFirst missing files:")
    for item in missing_train[:20]:
        print(item)

    raise FileNotFoundError(
        f"{len(missing_train)} training video files are missing."
    )


print("\nChecking validation video paths...")

missing_val = []

for p in val_df["video_path"]:
    path = resolve_video_path(p)

    if not path.exists():
        missing_val.append(str(path))

print(
    f"Validation files checked: {len(val_df)}"
)
print(
    f"Missing validation files: {len(missing_val)}"
)

if missing_val:
    print("\nFirst missing files:")
    for item in missing_val[:20]:
        print(item)

    raise FileNotFoundError(
        f"{len(missing_val)} validation video files are missing."
    )


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Deliberately no horizontal flip:
    # hand orientation can carry meaning in ISL.

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


val_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def sample_frames(
    video_path: Path,
    num_frames: int,
) -> list[Image.Image]:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"Video contains no readable frames: "
            f"{video_path}"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        num=num_frames,
        dtype=np.int64,
    )

    selected = [
        Image.fromarray(
            frames[int(i)]
        )
        for i in indices
    ]

    return selected


# ============================================================
# DATASET
# ============================================================

class SignVideoDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: dict[str, int],
        training: bool,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.class_to_idx = class_to_idx

        self.transform = (
            train_transform
            if training
            else val_transform
        )

        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        video_path = resolve_video_path(
            row["video_path"]
        )

        frames = sample_frames(
            video_path,
            NUM_FRAMES,
        )

        tensor_frames = []

        for frame in frames:
            tensor_frames.append(
                self.transform(frame)
            )

        video_tensor = torch.stack(
            tensor_frames,
            dim=0,
        )

        label = self.class_to_idx[
            str(row["gloss"])
        ]

        return (
            video_tensor,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================
# DATASETS / LOADERS
# ============================================================

train_dataset = SignVideoDataset(
    train_df,
    class_to_idx,
    training=True,
)

val_dataset = SignVideoDataset(
    val_df,
    class_to_idx,
    training=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# MODEL
# ============================================================

class MobileNetV3FullVocabulary(nn.Module):

    def __init__(
        self,
        num_classes: int,
    ):

        super().__init__()

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
        )

        base = models.mobilenet_v3_large(
            weights=weights
        )

        self.backbone = base.features

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.feature_dim = 960

        self.dropout = nn.Dropout(
            p=0.30
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.feature_dim,
                256,
            ),
            nn.Hardswish(),
            nn.Dropout(
                p=0.30
            ),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward_frames(
        self,
        frames: torch.Tensor,
    ):

        features = []

        for start in range(
            0,
            frames.shape[0],
            FRAME_CHUNK,
        ):

            chunk = frames[
                start:start + FRAME_CHUNK
            ]

            x = self.backbone(
                chunk
            )

            x = self.avgpool(x)

            x = torch.flatten(
                x,
                1,
            )

            features.append(x)

        return torch.cat(
            features,
            dim=0,
        )

    def forward(
        self,
        video: torch.Tensor,
    ):

        batch_size, num_frames, c, h, w = (
            video.shape
        )

        frames = video.reshape(
            batch_size * num_frames,
            c,
            h,
            w,
        )

        frame_features = (
            self.forward_frames(frames)
        )

        frame_features = frame_features.reshape(
            batch_size,
            num_frames,
            self.feature_dim,
        )

        # Temporal mean pooling.
        video_features = (
            frame_features.mean(dim=1)
        )

        video_features = self.dropout(
            video_features
        )

        logits = self.classifier(
            video_features
        )

        return logits, video_features


model = MobileNetV3FullVocabulary(
    NUM_CLASSES
).to(DEVICE)


# ============================================================
# FREEZE MOST BACKBONE LAYERS
# ============================================================

for parameter in model.backbone.parameters():
    parameter.requires_grad = False


# MobileNetV3-Large final blocks.
for block in list(model.backbone.children())[-4:]:
    for parameter in block.parameters():
        parameter.requires_grad = True


# ============================================================
# CLASS-BALANCED WEIGHTS
# ============================================================

counts = np.zeros(
    NUM_CLASSES,
    dtype=np.float32,
)

for gloss, count in (
    train_df["gloss"]
    .value_counts()
    .items()
):

    idx = class_to_idx[str(gloss)]
    counts[idx] = float(count)


# Inverse square-root weighting prevents
# extreme amplification of singleton classes.
weights = np.zeros_like(counts)

for i in range(NUM_CLASSES):

    if counts[i] > 0:
        weights[i] = (
            1.0 / np.sqrt(counts[i])
        )

mean_weight = weights[weights > 0].mean()

weights[weights > 0] /= mean_weight

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05,
)


# ============================================================
# OPTIMIZER
# ============================================================

backbone_parameters = [
    p
    for p in model.backbone.parameters()
    if p.requires_grad
]

classifier_parameters = list(
    model.classifier.parameters()
)

optimizer = torch.optim.AdamW(
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


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
)


# ============================================================
# AMP
# ============================================================

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # Macro F1 only over classes that actually
    # occur in this validation set.
    present_labels = sorted(
        set(y_true)
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return (
        float(accuracy),
        float(macro_f1),
        float(weighted_f1),
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    # Frozen backbone blocks should not update
    # BatchNorm running statistics.
    for block in list(model.backbone.children())[:-4]:
        block.eval()

    running_loss = 0.0
    all_true = []
    all_pred = []

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for videos, labels in progress:

        videos = videos.to(
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

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                logits, _ = model(videos)

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits, _ = model(videos)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

        predictions = (
            logits.argmax(dim=1)
        )

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        all_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss
        / len(train_dataset)
    )

    accuracy, macro_f1, weighted_f1 = (
        calculate_metrics(
            all_true,
            all_pred,
        )
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        weighted_f1,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    all_true = []
    all_pred = []

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    )

    for videos, labels in progress:

        videos = videos.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits, _ = model(videos)

                loss = criterion(
                    logits,
                    labels,
                )

        else:

            logits, _ = model(videos)

            loss = criterion(
                logits,
                labels,
            )

        predictions = (
            logits.argmax(dim=1)
        )

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        all_true.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    val_loss = (
        running_loss
        / len(val_dataset)
    )

    accuracy, macro_f1, weighted_f1 = (
        calculate_metrics(
            all_true,
            all_pred,
        )
    )

    return (
        val_loss,
        accuracy,
        macro_f1,
        weighted_f1,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    epoch,
    val_loss,
    val_accuracy,
    val_macro_f1,
    val_weighted_f1,
):

    torch.save(
        {
            "stage": "full_vocabulary_development",
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "val_macro_f1":
                val_macro_f1,

            "val_weighted_f1":
                val_weighted_f1,

            "class_to_idx":
                class_to_idx,

            "idx_to_class":
                idx_to_class,

            "num_classes":
                NUM_CLASSES,

            "num_frames":
                NUM_FRAMES,

            "feature_dim":
                model.feature_dim,

            "frame_chunk":
                FRAME_CHUNK,

            "backbone_lr":
                BACKBONE_LR,

            "classifier_lr":
                CLASSIFIER_LR,

            "image_size":
                IMAGE_SIZE,

            "seed":
                SEED,
        },
        BEST_CHECKPOINT,
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"Training videos: {len(train_dataset)}"
)

print(
    f"Validation videos: {len(val_dataset)}"
)

print(
    f"Classes: {NUM_CLASSES}"
)

print(
    f"Frames/video: {NUM_FRAMES}"
)

print(
    f"Feature dimension: {model.feature_dim}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Frame chunk: {FRAME_CHUNK}"
)

print()


print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


best_f1 = -1.0
epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    print()
    print(
        f"================ EPOCH "
        f"{epoch}/{MAX_EPOCHS} ================"
    )

    train_loss, train_acc, train_f1, train_wf1 = (
        train_one_epoch()
    )

    val_loss, val_acc, val_f1, val_wf1 = (
        validate()
    )

    scheduler.step(
        val_f1
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "train_weighted_f1": train_wf1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_weighted_f1": val_wf1,
        }
    )

    print()
    print(
        f"Train loss:       {train_loss:.4f}"
    )

    print(
        f"Train accuracy:   {train_acc * 100:.2f}%"
    )

    print(
        f"Train Macro-F1:   {train_f1 * 100:.2f}%"
    )

    print(
        f"Val loss:         {val_loss:.4f}"
    )

    print(
        f"Val accuracy:     {val_acc * 100:.2f}%"
    )

    print(
        f"Val Macro-F1:     {val_f1 * 100:.2f}%"
    )

    print(
        f"Val weighted-F1:  {val_wf1 * 100:.2f}%"
    )

    current_lr_backbone = (
        optimizer.param_groups[0]["lr"]
    )

    current_lr_classifier = (
        optimizer.param_groups[1]["lr"]
    )

    print(
        f"Backbone LR:      {current_lr_backbone:.2e}"
    )

    print(
        f"Classifier LR:    {current_lr_classifier:.2e}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        epochs_without_improvement = 0

        save_checkpoint(
            epoch=epoch,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            val_weighted_f1=val_wf1,
        )

        print()
        print(
            "NEW BEST CHECKPOINT SAVED"
        )
        print(
            BEST_CHECKPOINT
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = (
    ROOT
    / "models"
    / "evaluation"
    / "full_vocab_training_history.csv"
)

history_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

history_df.to_csv(
    history_path,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation Macro-F1: "
    f"{best_f1 * 100:.2f}%"
)

print(
    f"Best checkpoint:\n"
    f"{BEST_CHECKPOINT}"
)

print(
    f"Training history:\n"
    f"{history_path}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is the full-vocabulary development checkpoint."
)

print(
    "It uses all 4,764 classes and 7,049 labeled videos "
    "in the development split."
)

print(
    "The final production model will be trained after "
    "the development recipe is frozen."
)

print("=" * 70)