import os
import json
from collections import Counter


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


# ============================================================
# Read labels
# ============================================================

def get_labels(dataset_dir):

    labels = []

    json_files = sorted(
        f for f in os.listdir(dataset_dir)
        if f.endswith(".json")
    )

    for filename in json_files:

        path = os.path.join(
            dataset_dir,
            filename
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list) and len(data) == 1:
            labels.append(data[0])

    return labels


# ============================================================
# Load train / validation labels
# ============================================================

train_labels = get_labels(TRAIN_DIR)
val_labels = get_labels(VAL_DIR)

train_classes = set(train_labels)
val_classes = set(val_labels)

missing_from_train = val_classes - train_classes

train_counts = Counter(train_labels)
val_counts = Counter(val_labels)


# ============================================================
# Results
# ============================================================

print("======================================")
print("Person 2 Class Distribution Check")
print("======================================")

print()
print(
    f"Total classes in classes_final.json: "
    f"{len(class_to_idx)}"
)

print(
    f"Unique training classes: "
    f"{len(train_classes)}"
)

print(
    f"Unique validation classes: "
    f"{len(val_classes)}"
)

print(
    f"Validation classes missing from training: "
    f"{len(missing_from_train)}"
)

print()

# ------------------------------------------------------------
# Missing validation classes
# ------------------------------------------------------------

if missing_from_train:

    print("--------------------------------------")
    print("Validation classes missing from train")
    print("--------------------------------------")

    for label in sorted(missing_from_train):
        print(label)

else:

    print(
        "Every validation class also appears "
        "in the training set."
    )


# ============================================================
# Class frequency statistics
# ============================================================

print()
print("--------------------------------------")
print("Training class frequency")
print("--------------------------------------")

print(
    f"Minimum samples per class: "
    f"{min(train_counts.values())}"
)

print(
    f"Maximum samples per class: "
    f"{max(train_counts.values())}"
)

print(
    f"Average samples per class: "
    f"{len(train_labels) / len(train_classes):.2f}"
)


print()
print("--------------------------------------")
print("Validation class frequency")
print("--------------------------------------")

print(
    f"Minimum samples per class: "
    f"{min(val_counts.values())}"
)

print(
    f"Maximum samples per class: "
    f"{max(val_counts.values())}"
)

print(
    f"Average samples per class: "
    f"{len(val_labels) / len(val_classes):.2f}"
)


# ============================================================
# Most frequent training classes
# ============================================================

print()
print("--------------------------------------")
print("Top 20 training classes")
print("--------------------------------------")

for label, count in train_counts.most_common(20):

    print(
        f"{label}: {count}"
    )


# ============================================================
# Final
# ============================================================

print()
print("======================================")

if len(missing_from_train) == 0:

    print(
        "RESULT: All validation classes "
        "are represented in training."
    )

else:

    print(
        "RESULT: Some validation classes "
        "are missing from training."
    )

print("======================================")