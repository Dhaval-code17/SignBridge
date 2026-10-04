import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms

from models.mobilenet_model import MobileNetV3SignRecognizer


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CISLR_ROOT = ROOT / "data" / "raw" / "CISLR"

PROTOTYPE_CSV = CISLR_ROOT / "prototype.csv"
TEST_CSV = CISLR_ROOT / "test.csv"

VIDEO_ROOT = (
    CISLR_ROOT
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

CLASSES_JSON = ROOT / "configs" / "classes.json"

CHECKPOINT = (
    ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_stage2_best.pth"
)

OUTPUT_DIR = ROOT / "models" / "evaluation"

NUM_FRAMES = 16
IMAGE_SIZE = 224

OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("STAGE-2 TEMPORAL FEATURE MATCHING")
print("=" * 65)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# FILE CHECKS
# ============================================================

required = [
    PROTOTYPE_CSV,
    TEST_CSV,
    VIDEO_ROOT,
    CLASSES_JSON,
    CHECKPOINT,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CLASSES
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
    int(v): k
    for k, v in class_to_idx.items()
}

class_names = [
    idx_to_class[i]
    for i in range(num_classes)
]

allowed_classes = set(
    class_to_idx.keys()
)


# ============================================================
# LOAD DATA
# ============================================================

prototype_df = pd.read_csv(
    PROTOTYPE_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)

for df in [prototype_df, test_df]:

    df["uid"] = (
        df["uid"]
        .astype(str)
        .str.strip()
    )

    df["gloss"] = (
        df["gloss"]
        .astype(str)
        .str.strip()
    )


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
# VERIFY PROTOTYPES
# ============================================================

prototype_counts = (
    prototype_df["gloss"]
    .value_counts()
)

if len(prototype_df) != num_classes:
    raise ValueError(
        f"Expected {num_classes} prototypes, "
        f"got {len(prototype_df)}."
    )

if prototype_counts.min() != 1:
    raise ValueError(
        "At least one class has no prototype."
    )

if prototype_counts.max() != 1:
    raise ValueError(
        "At least one class has multiple prototypes."
    )


# ============================================================
# VERIFY PROTOTYPE/TEST SEPARATION
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
        "Prototype/test UID overlap detected."
    )


# ============================================================
# VIDEO PATHS
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


for label, df in [
    ("prototype", prototype_df),
    ("test", test_df),
]:

    missing = [
        str(p)
        for p in df["video_path"]
        if not p.exists()
    ]

    if missing:

        raise FileNotFoundError(
            f"{len(missing)} {label} videos are missing."
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
print(
    f"  Frames/video: {NUM_FRAMES}"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose(
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
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open:\n{video_path}"
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
            f"No frames decoded:\n{video_path}"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
    ).round().astype(int)

    return [
        frames[int(i)]
        for i in indices
    ]


# ============================================================
# LOAD STAGE-2 MODEL
# ============================================================

print(
    "\nLoading Stage-2 checkpoint..."
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

print(
    f"Checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Checkpoint validation Macro-F1: "
    f"{checkpoint['val_f1']:.4f}"
)

print(
    f"Checkpoint validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

model = MobileNetV3SignRecognizer(
    num_classes=num_classes,
    pretrained=False,
    freeze_backbone=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(
    f"Feature dimension: "
    f"{model.feature_dim}"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_sequence(
    video_path,
):

    frames = sample_video_frames(
        video_path,
        NUM_FRAMES,
    )

    tensors = [
        transform(frame)
        for frame in frames
    ]

    batch = torch.stack(
        tensors,
        dim=0,
    ).to(device)

    features = model.extract_features(
        batch
    )

    features = F.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu()


# ============================================================
# PROTOTYPE FEATURES
# ============================================================

print(
    "\nExtracting Stage-2 prototype features..."
)

prototype_sequences = {}

for row in prototype_df.itertuples():

    sequence = extract_feature_sequence(
        row.video_path
    )

    prototype_sequences[
        row.gloss
    ] = sequence

    print(
        f"  {row.gloss:15s} OK"
    )


# ============================================================
# MATCHING METHODS
# ============================================================

def mean_pool_score(
    query,
    prototype,
):

    q = F.normalize(
        query.mean(dim=0, keepdim=True),
        p=2,
        dim=1,
    )

    p = F.normalize(
        prototype.mean(dim=0, keepdim=True),
        p=2,
        dim=1,
    )

    return float(
        torch.sum(q * p).item()
    )


def diagonal_score(
    query,
    prototype,
):

    scores = torch.sum(
        query * prototype,
        dim=1,
    )

    return float(
        scores.mean().item()
    )


def best_shift_score(
    query,
    prototype,
):

    best = -1.0
    length = query.shape[0]

    for offset in OFFSETS:

        if offset >= 0:

            q = query[offset:length]
            p = prototype[:length - offset]

        else:

            shift = -offset

            q = query[:length - shift]
            p = prototype[shift:length]

        if q.shape[0] == 0:
            continue

        score = torch.sum(
            q * p,
            dim=1,
        ).mean().item()

        best = max(
            best,
            score,
        )

    return float(best)


def topk_pairwise_score(
    query,
    prototype,
    k=8,
):

    matrix = torch.matmul(
        query,
        prototype.T,
    )

    values = matrix.flatten()

    k = min(
        k,
        values.numel(),
    )

    return float(
        torch.topk(
            values,
            k=k,
        ).values.mean().item()
    )


# ============================================================
# EVALUATION
# ============================================================

methods = [
    "mean_pool",
    "diagonal",
    "best_shift",
    "topk_pairwise",
]

results = {
    method: {
        "targets": [],
        "predictions": [],
        "uids": [],
    }
    for method in methods
}


print(
    "\nEvaluating official test videos..."
)

for counter, row in enumerate(
    test_df.itertuples(),
    start=1,
):

    query = extract_feature_sequence(
        row.video_path
    )

    scores = {
        method: []
        for method in methods
    }

    for gloss in class_names:

        prototype = prototype_sequences[
            gloss
        ]

        scores["mean_pool"].append(
            mean_pool_score(
                query,
                prototype,
            )
        )

        scores["diagonal"].append(
            diagonal_score(
                query,
                prototype,
            )
        )

        scores["best_shift"].append(
            best_shift_score(
                query,
                prototype,
            )
        )

        scores["topk_pairwise"].append(
            topk_pairwise_score(
                query,
                prototype,
            )
        )

    target_idx = class_to_idx[
        row.gloss
    ]

    for method in methods:

        prediction_idx = int(
            np.argmax(
                scores[method]
            )
        )

        results[
            method
        ]["targets"].append(
            target_idx
        )

        results[
            method
        ]["predictions"].append(
            prediction_idx
        )

        results[
            method
        ]["uids"].append(
            row.uid
        )

    if counter % 25 == 0:

        print(
            f"  Processed "
            f"{counter}/{len(test_df)}"
        )


# ============================================================
# METRICS
# ============================================================

summary_rows = []

print(
    "\n" + "=" * 65
)

print(
    "STAGE-2 TEMPORAL MATCHING RESULTS"
)

print(
    "=" * 65
)

for method in methods:

    targets = results[
        method
    ]["targets"]

    predictions = results[
        method
    ]["predictions"]

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    summary_rows.append(
        {
            "method": method,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
    )

    print(
        f"{method:15s} "
        f"Accuracy={accuracy:.4f} "
        f"Macro-F1={macro_f1:.4f} "
        f"Weighted-F1={weighted_f1:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "stage2_temporal_matching_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


prediction_df = pd.DataFrame(
    {
        "uid": results["mean_pool"]["uids"],
        "true_gloss": [
            idx_to_class[i]
            for i in results[
                "mean_pool"
            ]["targets"]
        ],
    }
)

for method in methods:

    prediction_df[
        f"{method}_prediction"
    ] = [
        idx_to_class[i]
        for i in results[
            method
        ]["predictions"]
    ]

prediction_path = (
    OUTPUT_DIR
    / "stage2_temporal_matching_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


print(
    "\nSaved:"
)

print(
    f"  {summary_path}"
)

print(
    f"  {prediction_path}"
)

print(
    "\n" + "=" * 65
)

print(
    "STAGE-2 TEMPORAL MATCHING COMPLETE"
)

print(
    "=" * 65
)