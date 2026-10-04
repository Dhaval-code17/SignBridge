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

VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = r"C:\Users\nihav\Downloads\final_model1_features\configs\classes_final.json"

CHECKPOINT_PATH = "models/checkpoints/new_real_ctc_transformer.pth"


# ============================================================
# Exact duplicate validation files
# ============================================================

DUPLICATE_FILES = {
    "seq_00098.npy",
    "seq_00456.npy",
    "seq_00500.npy",
    "seq_00550.npy",
    "seq_00590.npy",
    "seq_01023.npy",
}


# ============================================================
# Load classes
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    classes_data = json.load(f)

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
    "cuda" if torch.cuda.is_available()
    else "cpu"
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
# Evaluate clean validation set
# ============================================================

total = 0
correct = 0
excluded = 0

print("======================================")
print("Clean Validation Evaluation")
print("======================================")

print(
    f"Device: {device}"
)

print(
    f"Original validation sequences: "
    f"{len([f for f in os.listdir(VAL_DIR) if f.endswith('.npy')])}"
)

print(
    f"Excluded exact duplicates: "
    f"{len(DUPLICATE_FILES)}"
)

print()


for feature_file in sorted(os.listdir(VAL_DIR)):

    if not feature_file.endswith(".npy"):
        continue

    if feature_file in DUPLICATE_FILES:

        excluded += 1
        continue

    feature_path = os.path.join(
        VAL_DIR,
        feature_file
    )

    label_path = os.path.join(
        VAL_DIR,
        feature_file.replace(
            ".npy",
            ".json"
        )
    )

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:
        expected = json.load(f)

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

    total += 1

    if predicted == expected:
        correct += 1


# ============================================================
# Results
# ============================================================

accuracy = (
    correct / total
) * 100


print("======================================")
print("Clean Validation Results")
print("======================================")

print(
    f"Sequences evaluated: "
    f"{total}"
)

print(
    f"Sequences excluded: "
    f"{excluded}"
)

print(
    f"Correct sequences: "
    f"{correct}"
)

print(
    f"Exact sequence accuracy: "
    f"{accuracy:.2f}%"
)

print("======================================")