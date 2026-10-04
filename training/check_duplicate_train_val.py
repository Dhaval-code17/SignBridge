import os
import numpy as np
from collections import defaultdict


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"


# ============================================================
# Exact sequence comparison
# ============================================================

def sequences_are_identical(a, b):

    if a.shape != b.shape:
        return False

    return np.array_equal(a, b)


# ============================================================
# Load training features
# ============================================================

print("======================================")
print("Train / Validation Duplicate Check")
print("======================================")
print()

print("Loading training features...")

train_features = {}

for filename in os.listdir(TRAIN_DIR):

    if not filename.endswith(".npy"):
        continue

    path = os.path.join(
        TRAIN_DIR,
        filename
    )

    train_features[filename] = np.load(path)


print(
    f"Training features loaded: "
    f"{len(train_features)}"
)

print()


# ============================================================
# Compare validation against training
# ============================================================

print("Checking validation features...")

exact_duplicates = []

zero_distance_matches = []

for val_filename in sorted(os.listdir(VAL_DIR)):

    if not val_filename.endswith(".npy"):
        continue

    val_path = os.path.join(
        VAL_DIR,
        val_filename
    )

    val_features = np.load(val_path)

    for train_filename, train_data in train_features.items():

        if train_data.shape != val_features.shape:
            continue

        if sequences_are_identical(
            train_data,
            val_features
        ):

            exact_duplicates.append(
                (
                    val_filename,
                    train_filename,
                    val_features.shape
                )
            )

            break


# ============================================================
# Results
# ============================================================

print()
print("======================================")
print("Duplicate Check Results")
print("======================================")

print(
    f"Exact train/validation duplicates: "
    f"{len(exact_duplicates)}"
)

print()

if exact_duplicates:

    print("Duplicate pairs:")

    for (
        val_filename,
        train_filename,
        shape
    ) in exact_duplicates:

        print(
            f"VAL {val_filename} "
            f"<-> TRAIN {train_filename} "
            f"shape={shape}"
        )

else:

    print(
        "No exact duplicate feature sequences found."
    )

print()
print("======================================")
print("Duplicate check complete.")
print("======================================")