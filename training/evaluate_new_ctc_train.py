import os
import sys
import json
import numpy as np
import torch

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
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("Training Set CTC Evaluation")
print("======================================")

print(f"Device: {device}")
print(f"Training directory: {TRAIN_DIR}")
print(f"Number of classes: {num_classes}")
print(f"CTC blank ID: {blank_id}")
print()


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

model.load_state_dict(
    checkpoint
)

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
# Evaluate
# ============================================================

feature_files = sorted(
    f
    for f in os.listdir(TRAIN_DIR)
    if f.endswith(".npy")
)

correct = 0
total = 0

sample_count = 0

for feature_file in feature_files:

    feature_path = os.path.join(
        TRAIN_DIR,
        feature_file
    )

    label_path = os.path.join(
        TRAIN_DIR,
        feature_file.replace(
            ".npy",
            ".json"
        )
    )

    features = np.load(
        feature_path
    )

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:
        expected = json.load(f)

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

    predicted = ctc_greedy_decode(
        logits
    )

    if predicted == expected:
        correct += 1

    total += 1

    if sample_count < 10:

        print(
            f"Sample {sample_count + 1}"
        )

        print(
            f"Expected: {expected}"
        )

        print(
            f"Predicted: {predicted}"
        )

        print()

        sample_count += 1


# ============================================================
# Results
# ============================================================

accuracy = (
    correct / total
) * 100

print("======================================")
print("Training Set Results")
print("======================================")

print(
    f"Training sequences: {total}"
)

print(
    f"Correct sequences: {correct}"
)

print(
    f"Exact sequence accuracy: {accuracy:.2f}%"
)

print("======================================")