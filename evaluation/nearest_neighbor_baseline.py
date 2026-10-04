import os
import json
import numpy as np


TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = (
    r"C:\Users\nihav\Downloads\final_model1_features"
    r"\configs\classes_final.json"
)


with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

class_to_idx = classes["class_to_idx"]


def load_sequences(data_dir):
    sequences = []

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".npy")
    )

    for filename in files:

        feature_path = os.path.join(
            data_dir,
            filename
        )

        label_path = os.path.join(
            data_dir,
            filename.replace(".npy", ".json")
        )

        features = np.load(feature_path).astype(
            np.float32
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:
            labels = json.load(f)

        label = labels[0]

        sequences.append(
            (features, label)
        )

    return sequences


def sequence_distance(a, b):
    """
    Compare two temporal feature sequences
    using mean-pooled feature vectors.
    """

    a = np.mean(a, axis=0)
    b = np.mean(b, axis=0)

    return np.mean(
        np.abs(a - b)
    )


print("=" * 50)
print("Nearest Neighbor Baseline")
print("=" * 50)

print("Loading training data...")

train_sequences = load_sequences(
    TRAIN_DIR
)

print(
    f"Training sequences: "
    f"{len(train_sequences)}"
)

print("Loading validation data...")

val_sequences = load_sequences(
    VAL_DIR
)

print(
    f"Validation sequences: "
    f"{len(val_sequences)}"
)

correct = 0

for index, (val_features, val_label) in enumerate(
    val_sequences
):

    best_distance = float("inf")
    best_label = None

    for train_features, train_label in train_sequences:

        distance = sequence_distance(
            val_features,
            train_features
        )

        if distance < best_distance:

            best_distance = distance
            best_label = train_label

    if best_label == val_label:
        correct += 1

    if (index + 1) % 100 == 0:

        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(val_sequences)}"
        )


accuracy = (
    correct /
    len(val_sequences) *
    100
)

print()
print("=" * 50)
print("Nearest Neighbor Results")
print("=" * 50)

print(
    f"Correct: {correct}"
)

print(
    f"Validation sequences: "
    f"{len(val_sequences)}"
)

print(
    f"Exact accuracy: "
    f"{accuracy:.2f}%"
)

print("=" * 50)