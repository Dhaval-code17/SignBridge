import os
import sys
import json
import numpy as np
import torch
from collections import Counter

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
# Evaluate validation set
# ============================================================

prediction_counts = Counter()

total = 0
empty_predictions = 0
correct = 0

for filename in sorted(os.listdir(VAL_DIR)):

    if not filename.endswith(".npy"):
        continue

    feature_path = os.path.join(
        VAL_DIR,
        filename
    )

    label_path = os.path.join(
        VAL_DIR,
        filename.replace(
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

    if len(predicted) == 0:

        empty_predictions += 1

    else:

        for label in predicted:
            prediction_counts[label] += 1


# ============================================================
# Results
# ============================================================

print("======================================")
print("Prediction Collapse Analysis")
print("======================================")

print(
    f"Validation sequences: {total}"
)

print(
    f"Correct sequences: {correct}"
)

print(
    f"Exact accuracy: "
    f"{(correct / total) * 100:.2f}%"
)

print(
    f"Empty predictions: "
    f"{empty_predictions}"
)

print(
    f"Non-empty predictions: "
    f"{total - empty_predictions}"
)

print()

print("======================================")
print("Unique Predicted Classes")
print("======================================")

print(
    f"Unique classes predicted: "
    f"{len(prediction_counts)}"
)

print(
    f"Total available classes: "
    f"{num_classes}"
)

print()

print("======================================")
print("Most Frequent Predictions")
print("======================================")

if prediction_counts:

    for label, count in prediction_counts.most_common(30):

        print(
            f"{label}: {count}"
        )

else:

    print("No non-empty predictions.")

print()

print("======================================")
print("Prediction Collapse Analysis Complete")
print("======================================")