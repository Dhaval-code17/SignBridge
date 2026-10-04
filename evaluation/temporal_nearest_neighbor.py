import os
import json
import numpy as np


TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = (
    r"C:\Users\nihav\Downloads\final_model1_features"
    r"\configs\classes_final.json"
)


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

        features = np.load(
            feature_path
        ).astype(np.float32)

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:
            labels = json.load(f)

        sequences.append(
            (
                features,
                labels[0]
            )
        )

    return sequences


def temporal_distance(a, b):
    """
    Compare two [T, 960] temporal sequences.

    Both sequences are resized to the same
    number of temporal positions using
    linear interpolation.
    """

    target_length = min(
        max(len(a), len(b)),
        64
    )

    def resize_sequence(x, new_length):

        old_length = len(x)

        if old_length == new_length:
            return x

        old_positions = np.linspace(
            0,
            1,
            old_length
        )

        new_positions = np.linspace(
            0,
            1,
            new_length
        )

        resized = np.empty(
            (new_length, x.shape[1]),
            dtype=np.float32
        )

        for d in range(x.shape[1]):

            resized[:, d] = np.interp(
                new_positions,
                old_positions,
                x[:, d]
            )

        return resized

    a = resize_sequence(
        a,
        target_length
    )

    b = resize_sequence(
        b,
        target_length
    )

    return np.mean(
        np.abs(a - b)
    )


print("=" * 50)
print("Temporal Nearest Neighbor Baseline")
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

for index, (
    val_features,
    val_label
) in enumerate(val_sequences):

    best_distance = float("inf")
    best_label = None

    for (
        train_features,
        train_label
    ) in train_sequences:

        distance = temporal_distance(
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
    len(val_sequences)
    * 100
)

print()
print("=" * 50)
print("Temporal Nearest Neighbor Results")
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