import os
import sys
import json

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
from torch.utils.data import DataLoader

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc
from models.ctc_transformer import CTCTransformer


# ============================================================
# Paths
# ============================================================

VAL_DIR = "data/continuous/new_val"

CLASSES_PATH = (
    r"C:\Users\nihav\Downloads\final_model1_features"
    r"\configs\classes_final.json"
)

CHECKPOINT_PATH = (
    "models/checkpoints/new_real_ctc_transformer.pth"
)


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load classes
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

idx_to_class = {
    int(k): v
    for k, v in class_data["idx_to_class"].items()
}

NUM_CLASSES = len(class_to_idx)
BLANK_ID = NUM_CLASSES


# ============================================================
# Dataset
# ============================================================

val_dataset = RealCTCDataset(
    VAL_DIR,
    class_to_idx
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_real_ctc
)


# ============================================================
# Model
# ============================================================

model = CTCTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=NUM_CLASSES,
    dropout=0.1,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


# ============================================================
# CTC Greedy Decoder
# ============================================================

def ctc_decode(logits):

    predictions = torch.argmax(
        logits,
        dim=-1
    )

    decoded_sequences = []

    for sequence in predictions:

        result = []
        previous = None

        for token in sequence.tolist():

            if token == BLANK_ID:
                previous = None
                continue

            if token != previous:
                result.append(token)

            previous = token

        decoded_sequences.append(result)

    return decoded_sequences


# ============================================================
# Edit Distance
# ============================================================

def edit_distance(reference, hypothesis):

    rows = len(reference) + 1
    cols = len(hypothesis) + 1

    dp = [
        [0] * cols
        for _ in range(rows)
    ]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):

        for j in range(1, cols):

            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


# ============================================================
# Evaluation
# ============================================================

total_sequences = 0
exact_matches = 0

total_edit_distance = 0
total_reference_tokens = 0

samples_printed = 0
MAX_SAMPLES = 10


with torch.no_grad():

    for (
        features,
        feature_lengths,
        targets,
        target_lengths
    ) in val_loader:

        features = features.to(DEVICE)

        logits = model(features)

        decoded = ctc_decode(logits)

        target_offset = 0

        for i in range(len(decoded)):

            target_length = target_lengths[i].item()

            reference = targets[
                target_offset:
                target_offset + target_length
            ].tolist()

            target_offset += target_length

            hypothesis = decoded[i]

            # ------------------------------------------------
            # Print first 10 samples
            # ------------------------------------------------

            if samples_printed < MAX_SAMPLES:

                reference_words = [
                    idx_to_class[token]
                    for token in reference
                ]

                hypothesis_words = [
                    idx_to_class[token]
                    for token in hypothesis
                    if token in idx_to_class
                ]

                print()
                print(
                    f"Sample {samples_printed + 1}"
                )

                print(
                    "Expected:",
                    reference_words
                )

                print(
                    "Predicted:",
                    hypothesis_words
                )

                samples_printed += 1

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            total_sequences += 1

            if reference == hypothesis:
                exact_matches += 1

            distance = edit_distance(
                reference,
                hypothesis
            )

            total_edit_distance += distance

            total_reference_tokens += len(reference)


# ============================================================
# Final Metrics
# ============================================================

exact_accuracy = (
    exact_matches / total_sequences
    if total_sequences > 0
    else 0.0
)

wer = (
    total_edit_distance / total_reference_tokens
    if total_reference_tokens > 0
    else 0.0
)


# ============================================================
# Results
# ============================================================

print()
print("======================================")
print("New CTC Transformer Evaluation")
print("======================================")

print(
    f"Validation sequences: "
    f"{total_sequences}"
)

print(
    f"Exact sequence accuracy: "
    f"{exact_accuracy * 100:.2f}%"
)

print(
    f"Token error rate: "
    f"{wer * 100:.2f}%"
)

print("======================================")