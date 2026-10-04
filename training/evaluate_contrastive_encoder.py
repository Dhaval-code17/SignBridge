from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "processed" / "final_dev_train.csv"
VAL_CSV = ROOT / "data" / "processed" / "final_dev_val.csv"
CLASSES_JSON = ROOT / "configs" / "classes_final.json"

CHECKPOINT = (
    ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_full_vocab_contrastive_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_FRAMES = 16
IMAGE_SIZE = 224
FRAME_CHUNK = 8

MAX_VAL = 200


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_df["gloss"] = (
    train_df["gloss"]
    .astype(str)
    .str.strip()
)

val_df["gloss"] = (
    val_df["gloss"]
    .astype(str)
    .str.strip()
)

val_df = (
    val_df
    .sort_values(["gloss", "uid"])
    .head(MAX_VAL)
    .reset_index(drop=True)
)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_video_path(value):
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
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ============================================================
# FRAME SAMPLING
# ============================================================

def sample_frames(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
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

    if not frames:
        raise RuntimeError(
            f"No readable frames: {video_path}"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        NUM_FRAMES,
        dtype=np.int64,
    )

    return [
        Image.fromarray(
            frames[int(index)]
        )
        for index in indices
    ]


# ============================================================
# ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        base = models.mobilenet_v3_large(
            weights=None
        )

        self.backbone = base.features

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

    def encode(self, video):

        batch_size, num_frames, channels, height, width = (
            video.shape
        )

        frames = video.reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )

        outputs = []

        for start in range(
            0,
            len(frames),
            FRAME_CHUNK,
        ):

            chunk = frames[
                start:start + FRAME_CHUNK
            ]

            x = self.backbone(chunk)

            x = self.avgpool(x)

            x = torch.flatten(
                x,
                1,
            )

            outputs.append(x)

        features = torch.cat(
            outputs,
            dim=0,
        )

        features = features.reshape(
            batch_size,
            num_frames,
            960,
        )

        clip_feature = features.mean(
            dim=1
        )

        return F.normalize(
            clip_feature.float(),
            dim=1,
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 70)
print("SIGNBRIDGE CONTRASTIVE ENCODER BENCHMARK")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Validation benchmark: {len(val_df)} videos")
print(f"Training references: {len(train_df)} videos")
print(f"Vocabulary: {len(class_to_idx)} classes")
print()

model = Encoder().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.backbone.load_state_dict(
    checkpoint["backbone_state_dict"]
)

model.eval()

print(
    f"Loaded checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation loss: "
    f"{checkpoint['val_loss']:.6f}"
)

print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(video_path):

    frames = sample_frames(
        video_path
    )

    tensor = torch.stack(
        [
            transform(frame)
            for frame in frames
        ]
    )

    tensor = tensor.unsqueeze(0).to(
        DEVICE
    )

    with torch.no_grad():

        feature = model.encode(
            tensor
        )

    return feature[0].cpu().numpy()


# ============================================================
# TRAINING REFERENCE EMBEDDINGS
# ============================================================

print(
    "Extracting training reference embeddings..."
)

train_features = []
train_labels = []

for _, row in tqdm(
    train_df.iterrows(),
    total=len(train_df),
    desc="Training references",
):

    feature = extract_feature(
        resolve_video_path(
            row["video_path"]
        )
    )

    train_features.append(feature)
    train_labels.append(
        row["gloss"]
    )


train_features = np.asarray(
    train_features,
    dtype=np.float32,
)

train_labels = np.asarray(
    train_labels
)


# ============================================================
# VALIDATION EMBEDDINGS
# ============================================================

print()
print(
    "Extracting validation embeddings..."
)

val_features = []
val_labels = []

for _, row in tqdm(
    val_df.iterrows(),
    total=len(val_df),
    desc="Validation",
):

    feature = extract_feature(
        resolve_video_path(
            row["video_path"]
        )
    )

    val_features.append(feature)
    val_labels.append(
        row["gloss"]
    )


val_features = np.asarray(
    val_features,
    dtype=np.float32,
)

val_labels = np.asarray(
    val_labels
)


# ============================================================
# NEAREST NEIGHBOUR
# ============================================================

print()
print(
    "Calculating nearest-neighbour predictions..."
)

similarity = (
    val_features
    @ train_features.T
)

nearest_indices = similarity.argmax(
    axis=1
)

pred_labels = train_labels[
    nearest_indices
]

top1_accuracy = float(
    np.mean(
        pred_labels == val_labels
    )
)


# ============================================================
# TOP-5
# ============================================================

top_k = 5

top_indices = np.argpartition(
    -similarity,
    kth=top_k - 1,
    axis=1,
)[:, :top_k]

top5_correct = np.array([
    val_labels[i]
    in train_labels[top_indices[i]]
    for i in range(
        len(val_labels)
    )
])

top5_accuracy = float(
    top5_correct.mean()
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

result_df = val_df[
    [
        "uid",
        "gloss",
        "video_path",
    ]
].copy()

result_df["predicted_gloss"] = (
    pred_labels
)

result_df["correct"] = (
    pred_labels == val_labels
)

result_path = (
    ROOT
    / "models"
    / "evaluation"
    / "contrastive_encoder_200_predictions.csv"
)

result_df.to_csv(
    result_path,
    index=False,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    ROOT
    / "models"
    / "evaluation"
    / "contrastive_encoder_200_summary.txt"
)

summary = (
    "SIGNBRIDGE CONTRASTIVE ENCODER BENCHMARK\n\n"
    f"Checkpoint epoch: {checkpoint['epoch']}\n"
    f"Checkpoint validation loss: {checkpoint['val_loss']:.6f}\n\n"
    f"Vocabulary classes: {len(class_to_idx)}\n"
    f"Training reference videos: {len(train_df)}\n"
    f"Validation benchmark videos: {len(val_df)}\n\n"
    f"Top-1 accuracy: {top1_accuracy * 100:.2f}%\n"
    f"Top-5 accuracy: {top5_accuracy * 100:.2f}%\n\n"
    "Feature dimension: 960\n"
    "Representation: mean pooled MobileNetV3 frame embeddings\n"
)

summary_path.write_text(
    summary,
    encoding="utf-8",
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Top-1 accuracy: "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Top-5 accuracy: "
    f"{top5_accuracy * 100:.2f}%"
)

print()
print(
    f"Predictions: {result_path}"
)

print(
    f"Summary: {summary_path}"
)

print("=" * 70)