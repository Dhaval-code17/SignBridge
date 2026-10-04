import os
import numpy as np
import torch

from models.ctc_transformer import CTCTransformer
from inference.ctc_decoder import CTCDecoder


# ------------------------------------------------
# Paths
# ------------------------------------------------

FEATURE_PATH = "data/person1/continuous_feature_sequence.npy"
CLASSES_PATH = "data/person1/classes.json"
MODEL_PATH = "models/checkpoints/ctc_transformer.pth"


# ------------------------------------------------
# Settings
# ------------------------------------------------

INPUT_DIM = 960
NUM_CLASSES = 39

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------
# Load features from Person 1
# ------------------------------------------------

print("Loading features...")

features = np.load(FEATURE_PATH)

print("Original feature shape:", features.shape)


# ------------------------------------------------
# Convert NumPy -> PyTorch
# ------------------------------------------------

features = torch.tensor(
    features,
    dtype=torch.float32
)


# Person 1 provides:
#
# [T, 960]
#
# The model expects:
#
# [B, T, 960]
#
# So add batch dimension.

features = features.unsqueeze(0)

features = features.to(DEVICE)

print("Model input shape:", features.shape)


# ------------------------------------------------
# Create CTC Transformer
# ------------------------------------------------

model = CTCTransformer(
    input_dim=INPUT_DIM,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=NUM_CLASSES,
    dropout=0.1
)


# ------------------------------------------------
# Load trained model
# ------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# ------------------------------------------------
# Load decoder
# ------------------------------------------------

decoder = CTCDecoder(
    CLASSES_PATH
)


# ------------------------------------------------
# Run inference
# ------------------------------------------------

print("\nRunning CTC inference...")

with torch.no_grad():

    logits = model(features)


# ------------------------------------------------
# Show model output
# ------------------------------------------------

print("Logits shape:", logits.shape)


# ------------------------------------------------
# Decode
# ------------------------------------------------

decoded_sequence = decoder.decode(
    logits
)


# ------------------------------------------------
# Final result
# ------------------------------------------------

print("\nRecognized sign sequence:")

for i, sequence in enumerate(decoded_sequence):

    print(f"Sample {i + 1}:")

    if len(sequence) == 0:
        print("  No signs detected")
    else:
        print(" ", sequence)