import os
import json
import numpy as np


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = (
    r"C:\Users\nihav\Downloads\final_model1_features"
    r"\configs\classes_final.json"
)


# ============================================================
# Load classes
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

print("======================================")
print("Person 2 Dataset Verification")
print("======================================")
print(
    f"Classes in classes_final.json: "
    f"{len(class_to_idx)}"
)
print()


# ============================================================
# Verify one dataset
# ============================================================

def verify_dataset(dataset_name, dataset_dir):

    print("--------------------------------------")
    print(dataset_name)
    print("--------------------------------------")

    feature_files = sorted(
        f for f in os.listdir(dataset_dir)
        if f.endswith(".npy")
    )

    errors = []

    checked = 0
    valid = 0

    for feature_file in feature_files:

        feature_path = os.path.join(
            dataset_dir,
            feature_file
        )

        label_file = feature_file.replace(
            ".npy",
            ".json"
        )

        label_path = os.path.join(
            dataset_dir,
            label_file
        )

        checked += 1

        # ----------------------------------------------------
        # Check label file
        # ----------------------------------------------------

        if not os.path.exists(label_path):

            errors.append(
                f"{feature_file}: missing label file"
            )

            continue

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                labels = json.load(f)

        except Exception as e:

            errors.append(
                f"{feature_file}: "
                f"could not read label JSON: {e}"
            )

            continue

        # ----------------------------------------------------
        # Check label format
        # ----------------------------------------------------

        if not isinstance(labels, list):

            errors.append(
                f"{feature_file}: "
                f"label is not a list"
            )

            continue

        if len(labels) != 1:

            errors.append(
                f"{feature_file}: "
                f"expected 1 label, found {len(labels)}"
            )

            continue

        label = labels[0]

        # ----------------------------------------------------
        # Check label exists in classes
        # ----------------------------------------------------

        if label not in class_to_idx:

            errors.append(
                f"{feature_file}: "
                f"unknown label '{label}'"
            )

            continue

        # ----------------------------------------------------
        # Check feature file
        # ----------------------------------------------------

        try:

            features = np.load(
                feature_path,
                mmap_mode="r"
            )

        except Exception as e:

            errors.append(
                f"{feature_file}: "
                f"could not load features: {e}"
            )

            continue

        # ----------------------------------------------------
        # Check feature shape
        # ----------------------------------------------------

        if features.ndim != 2:

            errors.append(
                f"{feature_file}: "
                f"expected 2D feature array, "
                f"got shape {features.shape}"
            )

            continue

        if features.shape[1] != 960:

            errors.append(
                f"{feature_file}: "
                f"expected [T, 960], "
                f"got {features.shape}"
            )

            continue

        # ----------------------------------------------------
        # Valid
        # ----------------------------------------------------

        valid += 1

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"Feature files: {len(feature_files)}"
    )

    print(
        f"Checked: {checked}"
    )

    print(
        f"Valid: {valid}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    if errors:

        print()
        print("First 20 errors:")

        for error in errors[:20]:
            print(" -", error)

    else:

        print("All feature/label pairs are valid.")

    print()

    return len(errors) == 0


# ============================================================
# Verify train and validation
# ============================================================

train_ok = verify_dataset(
    "TRAIN",
    TRAIN_DIR
)

val_ok = verify_dataset(
    "VALIDATION",
    VAL_DIR
)


# ============================================================
# Final result
# ============================================================

print("======================================")

if train_ok and val_ok:

    print(
        "RESULT: Dataset structure and "
        "feature-label pairing are valid."
    )

else:

    print(
        "RESULT: Dataset verification found "
        "problems."
    )

print("======================================")