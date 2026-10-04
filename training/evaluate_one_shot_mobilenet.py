import os
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torchvision import transforms

from models.mobilenet_model import MobileNetV3SignRecognizer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CISLR_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CISLR"
)

PROTOTYPE_CSV = CISLR_ROOT / "prototype.csv"
TEST_CSV = CISLR_ROOT / "test.csv"

VIDEO_ROOT = (
    CISLR_ROOT
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

CLASSES_JSON = (
    PROJECT_ROOT
    / "configs"
    / "classes.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "evaluation"
)

NUM_FRAMES = 16
IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("MOBILENETV3 ONE-SHOT CISLR BENCHMARK")
print("=" * 65)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    PROTOTYPE_CSV,
    TEST_CSV,
    VIDEO_ROOT,
    CLASSES_JSON,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(
    CLASSES_JSON,
    "r",
    encoding="utf-8",
) as f:

    class_config = json.load(f)


class_to_idx = class_config["class_to_idx"]

num_classes = class_config["num_classes"]

idx_to_class = {
    int(idx): name
    for name, idx in class_to_idx.items()
}

class_names = [
    idx_to_class[i]
    for i in range(num_classes)
]

allowed_classes = set(
    class_to_idx.keys()
)


# ============================================================
# LOAD CISLR CSV FILES
# ============================================================

prototype_df = pd.read_csv(
    PROTOTYPE_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)


prototype_df["uid"] = (
    prototype_df["uid"]
    .astype(str)
    .str.strip()
)

prototype_df["gloss"] = (
    prototype_df["gloss"]
    .astype(str)
    .str.strip()
)

test_df["uid"] = (
    test_df["uid"]
    .astype(str)
    .str.strip()
)

test_df["gloss"] = (
    test_df["gloss"]
    .astype(str)
    .str.strip()
)


# ============================================================
# FILTER TO 39 CLASSES
# ============================================================

prototype_df = prototype_df[
    prototype_df["gloss"].isin(
        allowed_classes
    )
].copy()

test_df = test_df[
    test_df["gloss"].isin(
        allowed_classes
    )
].copy()


# ============================================================
# VERIFY ONE PROTOTYPE PER CLASS
# ============================================================

prototype_counts = (
    prototype_df["gloss"]
    .value_counts()
)

if len(prototype_df) != num_classes:
    raise ValueError(
        "Expected exactly one prototype "
        f"per class. Got {len(prototype_df)}."
    )

if prototype_counts.min() != 1:
    raise ValueError(
        "Prototype set does not contain "
        "exactly one example per class."
    )

if prototype_counts.max() != 1:
    raise ValueError(
        "More than one prototype found "
        "for at least one class."
    )

if set(prototype_df["gloss"]) != allowed_classes:
    missing = sorted(
        allowed_classes
        - set(prototype_df["gloss"])
    )

    raise ValueError(
        f"Missing prototype classes: {missing}"
    )


# ============================================================
# VERIFY NO UID LEAKAGE
# ============================================================

prototype_uids = set(
    prototype_df["uid"]
)

test_uids = set(
    test_df["uid"]
)

overlap = prototype_uids & test_uids

if overlap:

    raise ValueError(
        "Prototype/test UID leakage detected:\n"
        + "\n".join(sorted(overlap))
    )


# ============================================================
# BUILD VIDEO PATHS
# ============================================================

prototype_df["video_path"] = prototype_df[
    "uid"
].apply(
    lambda uid: VIDEO_ROOT / f"{uid}.mp4"
)

test_df["video_path"] = test_df[
    "uid"
].apply(
    lambda uid: VIDEO_ROOT / f"{uid}.mp4"
)


# ============================================================
# VERIFY VIDEOS
# ============================================================

for name, df in [
    ("prototype", prototype_df),
    ("test", test_df),
]:

    missing = [
        str(path)
        for path in df["video_path"]
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            f"{len(missing)} {name} videos are missing."
        )


print("\nDataset:")
print(
    f"  Prototype videos: {len(prototype_df)}"
)
print(
    f"  Official test videos: {len(test_df)}"
)
print(
    f"  Classes: {num_classes}"
)


# ============================================================
# PREPROCESSING
# ============================================================

imagenet_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# VIDEO SAMPLING
# ============================================================

def sample_video_frames(
    video_path,
    num_frames=16,
):
    """
    Sequentially decode the complete video,
    then uniformly sample exactly num_frames.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
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
            f"No frames decoded:\n"
            f"{video_path}"
        )

    if len(frames) >= num_frames:

        indices = np.linspace(
            0,
            len(frames) - 1,
            num_frames,
        ).round().astype(int)

        sampled = [
            frames[i]
            for i in indices
        ]

    else:

        indices = np.linspace(
            0,
            len(frames) - 1,
            num_frames,
        ).round().astype(int)

        sampled = [
            frames[i]
            for i in indices
        ]

    return sampled


# ============================================================
# LOAD PRETRAINED MOBILENETV3
# ============================================================

print("\nLoading ImageNet-pretrained MobileNetV3...")

model = MobileNetV3SignRecognizer(
    num_classes=num_classes,
    pretrained=True,
    freeze_backbone=True,
)

model = model.to(device)

model.eval()

print(
    f"Feature dimension: "
    f"{model.feature_dim}"
)

if model.feature_dim != 960:

    raise ValueError(
        f"Expected feature dimension 960, "
        f"got {model.feature_dim}"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_video_embedding(
    video_path,
):

    frames = sample_video_frames(
        video_path,
        NUM_FRAMES,
    )

    tensors = []

    for frame in frames:

        tensor = imagenet_transform(
            frame
        )

        tensors.append(tensor)

    batch = torch.stack(
        tensors,
        dim=0,
    ).to(
        device,
        non_blocking=True,
    )

    features = model.extract_features(
        batch
    )

    # Average frame features
    video_feature = features.mean(
        dim=0,
        keepdim=True,
    )

    # L2 normalize
    video_feature = F.normalize(
        video_feature,
        p=2,
        dim=1,
    )

    return video_feature.squeeze(0).cpu()


# ============================================================
# EXTRACT PROTOTYPE FEATURES
# ============================================================

print("\nExtracting prototype features...")

prototype_embeddings = {}

for row in prototype_df.itertuples():

    gloss = row.gloss
    video_path = row.video_path

    embedding = extract_video_embedding(
        video_path
    )

    prototype_embeddings[
        gloss
    ] = embedding

    print(
        f"  Prototype: {gloss:15s} OK"
    )


prototype_matrix = torch.stack(
    [
        prototype_embeddings[
            class_name
        ]
        for class_name in class_names
    ],
    dim=0,
)


# ============================================================
# ONE-SHOT TEST PREDICTION
# ============================================================

print("\nEvaluating official CISLR test...")

all_targets = []
all_predictions = []
all_uids = []
all_similarities = []

for row in test_df.itertuples():

    embedding = extract_video_embedding(
        row.video_path
    )

    # Cosine similarity because both sides
    # are L2 normalized.
    similarities = torch.matmul(
        prototype_matrix,
        embedding,
    )

    predicted_idx = int(
        torch.argmax(
            similarities
        ).item()
    )

    predicted_gloss = idx_to_class[
        predicted_idx
    ]

    all_targets.append(
        class_to_idx[row.gloss]
    )

    all_predictions.append(
        predicted_idx
    )

    all_uids.append(
        row.uid
    )

    all_similarities.append(
        float(
            similarities[
                predicted_idx
            ].item()
        )
    )


# ============================================================
# METRICS
# ============================================================

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

weighted_f1 = f1_score(
    all_targets,
    all_predictions,
    average="weighted",
    zero_division=0,
)


print("\n" + "=" * 65)
print("LEAKAGE-FREE ONE-SHOT RESULTS")
print("=" * 65)

print(
    f"Accuracy   : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro-F1   : {macro_f1:.4f} "
    f"({macro_f1 * 100:.2f}%)"
)

print(
    f"Weighted-F1: {weighted_f1:.4f} "
    f"({weighted_f1 * 100:.2f}%)"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_targets,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(
    report
).transpose()

report_path = (
    OUTPUT_DIR
    / "one_shot_mobilenet_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(range(num_classes)),
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

cm_path = (
    OUTPUT_DIR
    / "one_shot_mobilenet_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# SAVE PER-VIDEO PREDICTIONS
# ============================================================

results_df = pd.DataFrame(
    {
        "uid": all_uids,
        "true_gloss": [
            idx_to_class[i]
            for i in all_targets
        ],
        "predicted_gloss": [
            idx_to_class[i]
            for i in all_predictions
        ],
        "similarity": all_similarities,
    }
)

results_path = (
    OUTPUT_DIR
    / "one_shot_mobilenet_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


# ============================================================
# TOP / BOTTOM CLASSES
# ============================================================

class_report = report_df.iloc[
    :num_classes
].copy()

class_report = class_report.sort_values(
    "f1-score",
    ascending=False,
)

print("\nTop 10 classes by F1:")

for class_name, row in class_report.head(10).iterrows():

    print(
        f"{class_name:15s} "
        f"Precision={row['precision']:.3f} "
        f"Recall={row['recall']:.3f} "
        f"F1={row['f1-score']:.3f}"
    )


print("\nBottom 10 classes by F1:")

for class_name, row in class_report.tail(10).iterrows():

    print(
        f"{class_name:15s} "
        f"Precision={row['precision']:.3f} "
        f"Recall={row['recall']:.3f} "
        f"F1={row['f1-score']:.3f}"
    )


# ============================================================
# SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "one_shot_mobilenet_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "Leakage-Free MobileNetV3 One-Shot CISLR Benchmark\n"
    )
    f.write("=" * 65 + "\n")
    f.write(
        f"Prototype samples: "
        f"{len(prototype_df)}\n"
    )
    f.write(
        f"Official test samples: "
        f"{len(test_df)}\n"
    )
    f.write(
        f"Classes: "
        f"{num_classes}\n"
    )
    f.write(
        f"Frames/video: "
        f"{NUM_FRAMES}\n"
    )
    f.write(
        f"Feature dimension: "
        f"{model.feature_dim}\n"
    )
    f.write(
        "Model weights: ImageNet pretrained "
        "MobileNetV3-Large\n"
    )
    f.write(
        "Classification method: "
        "Nearest prototype by cosine similarity\n"
    )
    f.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )
    f.write(
        f"Macro-F1: "
        f"{macro_f1:.6f}\n"
    )
    f.write(
        f"Weighted-F1: "
        f"{weighted_f1:.6f}\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved files:")

print(
    f"  {report_path}"
)

print(
    f"  {cm_path}"
)

print(
    f"  {results_path}"
)

print(
    f"  {summary_path}"
)

print("\n" + "=" * 65)
print("ONE-SHOT BENCHMARK COMPLETE")
print("=" * 65)