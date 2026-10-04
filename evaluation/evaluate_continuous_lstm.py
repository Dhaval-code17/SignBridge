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

from training.real_ctc_dataset import RealCTCDataset
from models.ctc_transformer import CTCTransformer


CHECKPOINT_PATH = "models/checkpoints/continuous_lstm.pth"
VAL_DIR = "data/continuous/val"
CLASSES_PATH = "data/person1/classes.json"

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


# Load classes
with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_config = json.load(f)

class_to_idx = class_config["class_to_idx"]
idx_to_class = {
    int(k): v
    for k, v in class_config["idx_to_class"].items()
}


# Load dataset
dataset = RealCTCDataset(
    VAL_DIR,
    class_to_idx
)


# Load model
model = ContinuousLSTM(
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )
)

model.eval()


def decode(logits, length):

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
            idx_to_class[index]
        )

        previous = index

    return decoded


correct_sequences = 0
total_sequences = len(dataset)

total_edit_distance = 0
total_true_signs = 0


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


with torch.no_grad():

    for index in range(total_sequences):

        features, target = dataset[index]

        features = features.unsqueeze(0)

        logits = model(features)[0]

        predicted = decode(
            logits,
            features.shape[1]
        )

        true_sequence = [
            idx_to_class[int(x)]
            for x in target.tolist()
        ]

        distance = edit_distance(
            true_sequence,
            predicted
        )

        total_edit_distance += distance
        total_true_signs += len(true_sequence)

        if predicted == true_sequence:
            correct_sequences += 1

        print(
            f"{index + 1}/{total_sequences} "
            f"True: {' '.join(true_sequence)} "
            f"-> Predicted: {' '.join(predicted)}"
        )


sequence_accuracy = (
    correct_sequences / total_sequences
) * 100

normalized_edit_distance = (
    total_edit_distance / total_true_signs
)


print()
print("===== Continuous LSTM Evaluation =====")
print(
    f"Validation sequences: {total_sequences}"
)
print(
    f"Exact sequence accuracy: "
    f"{sequence_accuracy:.2f}%"
)
print(
    f"Total edit distance: "
    f"{total_edit_distance}"
)
print(
    f"Normalized edit distance: "
    f"{normalized_edit_distance:.4f}"
)