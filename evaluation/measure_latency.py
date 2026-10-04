import sys
import os
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import numpy as np

from inference.continuous_recognizer import ContinuousRecognizer


VAL_DIR = "data/continuous/val"

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


# =========================
# Load Transformer
# =========================

transformer = ContinuousRecognizer()


# =========================
# Load LSTM
# =========================

lstm = ContinuousLSTM()

lstm.load_state_dict(
    torch.load(
        LSTM_CHECKPOINT,
        map_location="cpu"
    )
)

lstm.eval()


# =========================
# Collect validation files
# =========================

feature_files = sorted(
    [
        f
        for f in os.listdir(VAL_DIR)
        if f.endswith(".npy")
    ]
)


# =========================
# Measure Transformer
# =========================

transformer_times = []

for filename in feature_files:

    features = np.load(
        os.path.join(
            VAL_DIR,
            filename
        )
    )

    start = time.perf_counter()

    transformer.predict(features)

    elapsed = time.perf_counter() - start

    transformer_times.append(elapsed)


# =========================
# Measure LSTM
# =========================

lstm_times = []

with torch.no_grad():

    for filename in feature_files:

        features = np.load(
            os.path.join(
                VAL_DIR,
                filename
            )
        )

        features = torch.tensor(
            features,
            dtype=torch.float32
        ).unsqueeze(0)

        start = time.perf_counter()

        lstm(features)

        elapsed = time.perf_counter() - start

        lstm_times.append(elapsed)


# =========================
# Results
# =========================

transformer_average = (
    sum(transformer_times)
    / len(transformer_times)
)

lstm_average = (
    sum(lstm_times)
    / len(lstm_times)
)


print()
print("===== Latency Comparison =====")

print(
    f"Validation sequences: "
    f"{len(feature_files)}"
)

print(
    f"Transformer average latency: "
    f"{transformer_average * 1000:.2f} ms"
)

print(
    f"LSTM average latency: "
    f"{lstm_average * 1000:.2f} ms"
)