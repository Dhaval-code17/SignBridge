import os
import sys
import json
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.ctc_transformer import CTCTransformer


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = "data/continuous/new_train"
VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = r"C:\Users\nihav\Downloads\final_model1_features\configs\classes_final.json"

CHECKPOINT_PATH = "models/checkpoints/new_real_ctc_transformer.pth"


# ============================================================
# Load classes
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    classes_data = json.load(f)

class_to_idx = classes_data["class_to_idx"]
idx_to_class = {
    int(k): v
    for k, v in classes_data["idx_to_class"].items()
}

num_classes = classes_data["num_classes"]
blank_id = num_classes


# ============================================================
# Load labels
# ============================================================

def load_labels(dataset_dir):

    labels = {}

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

            feature_file = filename.replace(
                ".json",
                ".npy"
            )

            labels[feature_file] = data[0]

    return labels


train_labels = load_labels(TRAIN_DIR)
val_labels = load_labels(VAL_DIR)

train_counts = Counter(
    train_labels.values()
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model
# ============================================================

model = CTCTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=num_classes,
    dropout=0.1,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# ============================================================
# CTC greedy decoding
# ============================================================

def ctc_greedy_decode(logits):

    predictions = torch.argmax(
        logits,
        dim=-1
    )[0]

    decoded = []

    previous = None

    for token in predictions.tolist():

        if token == blank_id:

            previous = None
            continue

        if token == previous:
            continue

        decoded.append(
            idx_to_class[token]
        )

        previous = token

    return decoded


# ============================================================
# Result groups
# ============================================================

groups = {
    "1": {
        "correct": 0,
        "total": 0,
    },
    "2": {
        "correct": 0,
        "total": 0,
    },
    "3+": {
        "correct": 0,
        "total": 0,
    },
}


# ============================================================
# Evaluate validation set
# ============================================================

print("======================================")
print("Validation Accuracy By Class Support")
print("======================================")

print(
    f"Device: {device}"
)

print(
    f"Validation sequences: {len(val_labels)}"
)

print()


for feature_file in sorted(val_labels):

    expected_label = val_labels[feature_file]

    support = train_counts[expected_label]

    if support == 1:
        group = "1"

    elif support == 2:
        group = "2"

    else:
        group = "3+"

    feature_path = os.path.join(
        VAL_DIR,
        feature_file
    )

    features = np.load(
        feature_path
    )

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

    predicted = ctc_greedy_decode(
        logits
    )

    expected = [expected_label]

    groups[group]["total"] += 1

    if predicted == expected:

        groups[group]["correct"] += 1


# ============================================================
# Results
# ============================================================

print("======================================")
print("Results")
print("======================================")

for group in ["1", "2", "3+"]:

    correct = groups[group]["correct"]
    total = groups[group]["total"]

    if total > 0:

        accuracy = (
            correct / total
        ) * 100

    else:

        accuracy = 0.0

    print(
        f"{group} training sample(s): "
        f"{correct}/{total} correct "
        f"({accuracy:.2f}%)"
    )


# ============================================================
# Overall
# ============================================================

total_correct = sum(
    item["correct"]
    for item in groups.values()
)

total_samples = sum(
    item["total"]
    for item in groups.values()
)

overall_accuracy = (
    total_correct / total_samples
) * 100


print()

print("======================================")
print("Overall")
print("======================================")

print(
    f"Correct: "
    f"{total_correct}/{total_samples}"
)

print(
    f"Exact sequence accuracy: "
    f"{overall_accuracy:.2f}%"
)

print("======================================")