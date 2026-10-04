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
# Load feature statistics
# ============================================================

def get_feature_stats(dataset_dir, label_map):

    class_stats = defaultdict(list)

    feature_files = sorted(
        f for f in os.listdir(dataset_dir)
        if f.endswith(".npy")
    )

    for feature_file in feature_files:

        if feature_file not in label_map:
            continue

        path = os.path.join(
            dataset_dir,
            feature_file
        )

        features = np.load(path)

        class_name = label_map[feature_file]

        stats = {
            "frames": features.shape[0],
            "mean": float(features.mean()),
            "std": float(features.std()),
            "min": float(features.min()),
            "max": float(features.max()),
        }

        class_stats[class_name].append(stats)

    return class_stats


# ============================================================
# Load data
# ============================================================

print("======================================")
print("Person 2 Feature Distribution Check")
print("======================================")
print()

print("Loading training labels...")
train_labels = load_label_map(TRAIN_DIR)

print("Loading validation labels...")
val_labels = load_label_map(VAL_DIR)

print(
    f"Training sequences: {len(train_labels)}"
)

print(
    f"Validation sequences: {len(val_labels)}"
)

print()


print("Loading training feature statistics...")
train_stats = get_feature_stats(
    TRAIN_DIR,
    train_labels
)

print("Loading validation feature statistics...")
val_stats = get_feature_stats(
    VAL_DIR,
    val_labels
)

print()


# ============================================================
# Compare matching classes
# ============================================================

mean_differences = []
std_differences = []
frame_differences = []

compared_classes = 0

for class_name in sorted(val_stats):

    if class_name not in train_stats:
        continue

    train_items = train_stats[class_name]
    val_items = val_stats[class_name]

    # Validation has one sample per class
    val_item = val_items[0]

    # Compare validation sample against
    # all training samples of the same class

    for train_item in train_items:

        mean_diff = abs(
            train_item["mean"]
            - val_item["mean"]
        )

        std_diff = abs(
            train_item["std"]
            - val_item["std"]
        )

        frame_diff = abs(
            train_item["frames"]
            - val_item["frames"]
        )

        mean_differences.append(mean_diff)
        std_differences.append(std_diff)
        frame_differences.append(frame_diff)

    compared_classes += 1


# ============================================================
# Results
# ============================================================

print("======================================")
print("Comparison Results")
print("======================================")

print(
    f"Classes compared: "
    f"{compared_classes}"
)

print()

if mean_differences:

    print(
        "Mean absolute feature-mean difference:"
    )

    print(
        f"{np.mean(mean_differences):.6f}"
    )

    print()

    print(
        "Mean absolute feature-std difference:"
    )

    print(
        f"{np.mean(std_differences):.6f}"
    )

    print()

    print(
        "Average frame-count difference:"
    )

    print(
        f"{np.mean(frame_differences):.2f}"
    )

    print()

    print(
        "Maximum feature-mean difference:"
    )

    print(
        f"{np.max(mean_differences):.6f}"
    )

    print()

    print(
        "Maximum feature-std difference:"
    )

    print(
        f"{np.max(std_differences):.6f}"
    )

    print()

    print(
        "Maximum frame-count difference:"
    )

    print(
        f"{np.max(frame_differences):.0f}"
    )

else:

    print(
        "No matching classes were found."
    )


# ============================================================
# Overall feature statistics
# ============================================================

print()
print("======================================")
print("Overall Feature Statistics")
print("======================================")

train_means = []
train_stds = []
train_lengths = []

for class_items in train_stats.values():

    for item in class_items:

        train_means.append(item["mean"])
        train_stds.append(item["std"])
        train_lengths.append(item["frames"])


val_means = []
val_stds = []
val_lengths = []

for class_items in val_stats.values():

    for item in class_items:

        val_means.append(item["mean"])
        val_stds.append(item["std"])
        val_lengths.append(item["frames"])


print()
print("TRAIN")
print(
    f"Mean of sequence means: "
    f"{np.mean(train_means):.6f}"
)
print(
    f"Mean of sequence stds: "
    f"{np.mean(train_stds):.6f}"
)
print(
    f"Average sequence length: "
    f"{np.mean(train_lengths):.2f}"
)

print()
print("VALIDATION")
print(
    f"Mean of sequence means: "
    f"{np.mean(val_means):.6f}"
)
print(
    f"Mean of sequence stds: "
    f"{np.mean(val_stds):.6f}"
)
print(
    f"Average sequence length: "
    f"{np.mean(val_lengths):.2f}"
)

print()
print("======================================")
print("Feature distribution check complete.")
print("======================================")