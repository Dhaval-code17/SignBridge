import os
import json
import numpy as np
from collections import defaultdict


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"


# ============================================================
# Load labels
# ============================================================

def load_label_map(dataset_dir):

    label_map = {}

    for filename in os.listdir(dataset_dir):

        if not filename.endswith(".json"):
            continue

        path = os.path.join(
            dataset_dir,
            filename
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            labels = json.load(f)

        if isinstance(labels, list) and len(labels) == 1:

            feature_name = filename.replace(
                ".json",
                ".npy"
            )

            label_map[feature_name] = labels[0]

    return label_map


# ============================================================
# Resample sequence
# ============================================================

def resample_sequence(features, target_length):

    if features.shape[0] == target_length:
        return features

    old_positions = np.linspace(
        0.0,
        1.0,
        features.shape[0]
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        target_length
    )

    resampled = np.empty(
        (
            target_length,
            features.shape[1]
        ),
        dtype=np.float32
    )

    for feature_index in range(
        features.shape[1]
    ):

        resampled[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            features[:, feature_index]
        )

    return resampled


# ============================================================
# Sequence distance
# ============================================================

def sequence_distance(
    validation_features,
    training_features
):

    training_resampled = resample_sequence(
        training_features,
        validation_features.shape[0]
    )

    difference = (
        validation_features
        - training_resampled
    )

    return float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )


# ============================================================
# Load labels
# ============================================================

print("======================================")
print("Same-Class Sequence Distance Analysis")
print("======================================")
print()

print("Loading training labels...")
train_labels = load_label_map(
    TRAIN_DIR
)

print("Loading validation labels...")
val_labels = load_label_map(
    VAL_DIR
)

print(
    f"Training sequences: "
    f"{len(train_labels)}"
)

print(
    f"Validation sequences: "
    f"{len(val_labels)}"
)

print()


# ============================================================
# Group training files by class
# ============================================================

train_by_class = defaultdict(list)

for feature_file, label in train_labels.items():

    train_by_class[label].append(
        feature_file
    )


# ============================================================
# Compare validation against same-class training
# ============================================================

best_distances = []
average_distances = []
worst_distances = []

comparison_records = []

for val_file, val_label in val_labels.items():

    if val_label not in train_by_class:
        continue

    val_path = os.path.join(
        VAL_DIR,
        val_file
    )

    val_features = np.load(
        val_path
    )

    distances = []

    for train_file in train_by_class[val_label]:

        train_path = os.path.join(
            TRAIN_DIR,
            train_file
        )

        train_features = np.load(
            train_path
        )

        distance = sequence_distance(
            val_features,
            train_features
        )

        distances.append(distance)

    if not distances:
        continue

    best_distance = min(distances)
    average_distance = np.mean(distances)
    worst_distance = max(distances)

    best_distances.append(
        best_distance
    )

    average_distances.append(
        average_distance
    )

    worst_distances.append(
        worst_distance
    )

    comparison_records.append(
        (
            val_label,
            val_file,
            len(distances),
            best_distance,
            average_distance,
            worst_distance
        )
    )


# ============================================================
# Results
# ============================================================

print("======================================")
print("Distance Results")
print("======================================")

print(
    f"Validation classes compared: "
    f"{len(comparison_records)}"
)

print()

if best_distances:

    print(
        "Average BEST same-class distance:"
    )

    print(
        f"{np.mean(best_distances):.6f}"
    )

    print()

    print(
        "Average MEAN same-class distance:"
    )

    print(
        f"{np.mean(average_distances):.6f}"
    )

    print()

    print(
        "Average WORST same-class distance:"
    )

    print(
        f"{np.mean(worst_distances):.6f}"
    )

    print()

    print(
        "Maximum BEST same-class distance:"
    )

    print(
        f"{np.max(best_distances):.6f}"
    )

    print()

    print(
        "Minimum BEST same-class distance:"
    )

    print(
        f"{np.min(best_distances):.6f}"
    )


# ============================================================
# Closest validation sequences
# ============================================================

print()
print("======================================")
print("Closest Validation Sequences")
print("======================================")

closest = sorted(
    comparison_records,
    key=lambda x: x[3]
)[:20]

for (
    label,
    filename,
    count,
    best_distance,
    average_distance,
    worst_distance
) in closest:

    print(
        f"{label} | "
        f"train samples={count} | "
        f"best={best_distance:.6f} | "
        f"avg={average_distance:.6f}"
    )


# ============================================================
# Farthest validation sequences
# ============================================================

print()
print("======================================")
print("Farthest Validation Sequences")
print("======================================")

farthest = sorted(
    comparison_records,
    key=lambda x: x[3],
    reverse=True
)[:20]

for (
    label,
    filename,
    count,
    best_distance,
    average_distance,
    worst_distance
) in farthest:

    print(
        f"{label} | "
        f"train samples={count} | "
        f"best={best_distance:.6f} | "
        f"avg={average_distance:.6f}"
    )


print()
print("======================================")
print("Analysis complete.")
print("======================================")