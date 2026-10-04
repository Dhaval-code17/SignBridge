import os
import json
from collections import Counter, defaultdict


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"


# ============================================================
# Load labels
# ============================================================

def load_labels(dataset_dir):

    labels = []

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
            data = json.load(f)

        if isinstance(data, list) and len(data) == 1:
            labels.append(data[0])

    return labels


# ============================================================
# Load
# ============================================================

print("======================================")
print("Validation Class Support Analysis")
print("======================================")
print()

train_labels = load_labels(TRAIN_DIR)
val_labels = load_labels(VAL_DIR)

print(
    f"Training sequences: {len(train_labels)}"
)

print(
    f"Validation sequences: {len(val_labels)}"
)

print()


# ============================================================
# Count training examples per class
# ============================================================

train_counts = Counter(train_labels)

val_classes = sorted(set(val_labels))


# ============================================================
# Group validation classes by training support
# ============================================================

support_groups = defaultdict(list)

for class_name in val_classes:

    count = train_counts[class_name]

    support_groups[count].append(
        class_name
    )


# ============================================================
# Results
# ============================================================

print("======================================")
print("Training Support For Validation Classes")
print("======================================")
print()

for support in sorted(support_groups):

    classes = support_groups[support]

    print(
        f"{support} training sample(s): "
        f"{len(classes)} validation classes"
    )

print()


# ============================================================
# Summary
# ============================================================

single = len(
    support_groups.get(1, [])
)

double = len(
    support_groups.get(2, [])
)

triple_or_more = sum(
    len(classes)
    for support, classes in support_groups.items()
    if support >= 3
)


print("======================================")
print("Summary")
print("======================================")

print(
    f"Validation classes with 1 training sample: "
    f"{single}"
)

print(
    f"Validation classes with 2 training samples: "
    f"{double}"
)

print(
    f"Validation classes with 3+ training samples: "
    f"{triple_or_more}"
)

print()


# ============================================================
# Lowest-support examples
# ============================================================

print("======================================")
print("Lowest-Support Validation Classes")
print("======================================")

for class_name in val_classes[:30]:

    print(
        f"{class_name}: "
        f"{train_counts[class_name]} training sample(s)"
    )

print()
print("======================================")
print("Analysis complete.")
print("======================================")