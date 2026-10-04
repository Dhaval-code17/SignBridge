import sys
import os
import json

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import numpy as np

from training.real_ctc_dataset import RealCTCDataset
from inference.continuous_recognizer import ContinuousRecognizer


VAL_DIR = "data/continuous/val"
CLASSES_PATH = "data/person1/classes.json"


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
                dp[i][j] = dp[i - 1][j - 1]

            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    return dp[-1][-1]


with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_config = json.load(f)

class_to_idx = class_config["class_to_idx"]


dataset = RealCTCDataset(
    VAL_DIR,
    class_to_idx
)


# =========================
# Transformer WER
# =========================

transformer = ContinuousRecognizer()

transformer_errors = 0
transformer_words = 0

for index in range(len(dataset)):

    features, target = dataset[index]

    true_sequence = [
        class_config["idx_to_class"][str(int(x))]
        for x in target.tolist()
    ]

    predicted_sequence = transformer.predict(
        features.numpy()
    )

    transformer_errors += edit_distance(
        true_sequence,
        predicted_sequence
    )

    transformer_words += len(true_sequence)


transformer_wer = (
    transformer_errors / transformer_words
)


# =========================
# LSTM WER
# =========================

LSTM_CHECKPOINT = (
    "models/checkpoints/continuous_lstm.pth"
)

INPUT_DIM = 960
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 39
BLANK_ID = 39


class ContinuousLSTM(torch.nn.Module):

    def __init__(
        self,
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=39
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = torch.nn.Linear(
            hidden_dim,
            num_classes + 1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        return self.classifier(output)


lstm = ContinuousLSTM()

lstm.load_state_dict(
    torch.load(
        LSTM_CHECKPOINT,
        map_location="cpu"
    )
)

lstm.eval()


def decode_lstm(logits, length):

    predictions = torch.argmax(
        logits[:length],
        dim=-1
    )

    decoded = []

    previous = None

    for index in predictions.tolist():

        if index == BLANK_ID:
            previous = None
            continue

        if index == previous:
            continue

        decoded.append(
            class_config["idx_to_class"][str(index)]
        )

        previous = index

    return decoded


lstm_errors = 0
lstm_words = 0


with torch.no_grad():

    for index in range(len(dataset)):

        features, target = dataset[index]

        input_features = features.unsqueeze(0)

        logits = lstm(
            input_features
        )[0]

        predicted_sequence = decode_lstm(
            logits,
            features.shape[0]
        )

        true_sequence = [
            class_config["idx_to_class"][str(int(x))]
            for x in target.tolist()
        ]

        lstm_errors += edit_distance(
            true_sequence,
            predicted_sequence
        )

        lstm_words += len(true_sequence)


lstm_wer = (
    lstm_errors / lstm_words
)


# =========================
# Results
# =========================

print()
print("===== WER Comparison =====")

print(
    f"LSTM WER: "
    f"{lstm_wer:.4f} "
    f"({lstm_wer * 100:.2f}%)"
)

print(
    f"Transformer WER: "
    f"{transformer_wer:.4f} "
    f"({transformer_wer * 100:.2f}%)"
)