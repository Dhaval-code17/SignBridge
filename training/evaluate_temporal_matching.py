import os
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CISLR_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "CISLR"
)

PROTOTYPE_CSV = CISLR_ROOT / "prototype.csv"
TEST_CSV = CISLR_ROOT / "test.csv"

VIDEO_ROOT = (
    CISLR_ROOT
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

CLASSES_JSON = (
    PROJECT_ROOT / "configs" / "classes.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "evaluation"
)

NUM_FRAMES = 16
IMAGE_SIZE = 224

# Temporal offsets tested by aligned matching.
OFFSETS = [-3, -2, -1, 0, 1, 2, 3]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("MOBILENETV3 TEMPORAL FEATURE MATCHING BENCHMARK")
print("=" * 65)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# FILE CHECKS
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
# CLASS MAPPING
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

counts = (
    prototype_df["gloss"]
    .value_counts()
)

if len(prototype_df) != num_classes:
    raise ValueError(
        f"Expected {num_classes} prototypes, "
        f"got {len(prototype_df)}"
    )

if counts.min() != 1 or counts.max() != 1:
    raise ValueError(
        "Expected exactly one prototype per class."
    )


# ============================================================
# VERIFY NO PROTOTYPE/TEST UID OVERLAP
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
        "Prototype/test leakage detected."
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
            f"{len(missing)} {name} videos missing."
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
# IMAGE PREPROCESSING
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

    if not frames:
        raise RuntimeError(
            f"No frames decoded:\n"
            f"{video_path}"
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
# MODEL
# ============================================================

print(
    "\nLoading ImageNet-pretrained "
    "MobileNetV3-Large..."
)

model = MobileNetV3SignRecognizer(
    num_classes=num_classes,
    pretrained=True,
    freeze_backbone=True,
)

model = model.to(device)
model.eval()

print(
    f"Feature dimension: {model.feature_dim}"
)

if model.feature_dim != 960:
    raise ValueError(
        f"Expected 960 features, "
        f"got {model.feature_dim}"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_sequence(video_path):
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

    # L2 normalize each frame independently.
    features = F.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu()


# ============================================================
# EXTRACT PROTOTYPE SEQUENCES
# ============================================================

print(
    "\nExtracting prototype feature sequences..."
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
# TEMPORAL SCORING FUNCTIONS
# ============================================================

def mean_pool_score(
    query,
    prototype,
):
    """
    Current baseline:
    average frames first, then cosine similarity.
    """

    query_mean = F.normalize(
        query.mean(dim=0, keepdim=True),
        p=2,
        dim=1,
    )

    prototype_mean = F.normalize(
        prototype.mean(dim=0, keepdim=True),
        p=2,
        dim=1,
    )

    return float(
        torch.sum(
            query_mean * prototype_mean
        ).item()
    )


def diagonal_score(
    query,
    prototype,
):
    """
    Same-time-step cosine similarity.
    """

    similarities = torch.sum(
        query * prototype,
        dim=1,
    )

    return float(
        similarities.mean().item()
    )


def best_shift_score(
    query,
    prototype,
):
    """
    Compare the sequences at small temporal offsets.
    Uses the best average aligned similarity.
    """

    best_score = -1.0

    length = query.shape[0]

    for offset in OFFSETS:

        if offset >= 0:

            q = query[
                offset:length
            ]

            p = prototype[
                :length - offset
            ]

        else:

            shift = -offset

            q = query[
                :length - shift
            ]

            p = prototype[
                shift:length
            ]

        if len(q) == 0:
            continue

        score = torch.sum(
            q * p,
            dim=1,
        ).mean().item()

        if score > best_score:
            best_score = score

    return float(best_score)


def topk_pairwise_score(
    query,
    prototype,
    k=8,
):
    """
    Pairwise frame similarity matrix.
    Take the strongest k frame-pairs.
    """

    similarity_matrix = torch.matmul(
        query,
        prototype.T,
    )

    flattened = similarity_matrix.flatten()

    k = min(
        k,
        flattened.numel(),
    )

    top_values = torch.topk(
        flattened,
        k=k,
    ).values

    return float(
        top_values.mean().item()
    )


# ============================================================
# EVALUATION STORAGE
# ============================================================

methods = [
    "mean_pool",
    "diagonal",
    "best_shift",
    "topk_pairwise",
]

all_results = {
    method: {
        "targets": [],
        "predictions": [],
        "uids": [],
    }
    for method in methods
}


# ============================================================
# TEST EVALUATION
# ============================================================

print(
    "\nEvaluating official CISLR test..."
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

    true_idx = class_to_idx[
        row.gloss
    ]

    for method in methods:

        predicted_idx = int(
            np.argmax(
                scores[method]
            )
        )

        all_results[
            method
        ]["targets"].append(
            true_idx
        )

        all_results[
            method
        ]["predictions"].append(
            predicted_idx
        )

        all_results[
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
    "TEMPORAL MATCHING RESULTS"
)

print(
    "=" * 65
)

for method in methods:

    targets = all_results[
        method
    ]["targets"]

    predictions = all_results[
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
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "temporal_matching_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "uid": all_results[
            "mean_pool"
        ]["uids"],
        "true_gloss": [
            idx_to_class[i]
            for i in all_results[
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
        for i in all_results[
            method
        ]["predictions"]
    ]

predictions_path = (
    OUTPUT_DIR
    / "temporal_matching_predictions.csv"
)

prediction_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nSaved:"
)

print(
    f"  {summary_path}"
)

print(
    f"  {predictions_path}"
)

print(
    "\n" + "=" * 65
)

print(
    "TEMPORAL MATCHING BENCHMARK COMPLETE"
)

print(
    "=" * 65
)