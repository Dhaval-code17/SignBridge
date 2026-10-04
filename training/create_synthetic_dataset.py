import os
import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

NUM_SAMPLES = 1000
SEQUENCE_LENGTH = 16
FEATURE_DIM = 960
NUM_CLASSES = 39

OUTPUT_DIR = "data/synthetic"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# The same 39 class names supplied by Person 1
CLASSES = [
    "angry",
    "ant",
    "august",
    "bad",
    "bill",
    "brain",
    "cat",
    "christmas",
    "date",
    "duck",
    "friday",
    "goat",
    "good",
    "grow",
    "hot",
    "increase",
    "india",
    "interest",
    "iron",
    "land",
    "leaf",
    "light",
    "monday",
    "month",
    "mother",
    "name",
    "pencil",
    "reservation",
    "rubber",
    "saturday",
    "sick",
    "sunday",
    "thank you",
    "thursday",
    "tuesday",
    "wash",
    "wednesday",
    "week",
    "yellow",
]

# -----------------------------
# Generate synthetic data
# -----------------------------

rows = []

for i in range(NUM_SAMPLES):

    # Shape:
    # 16 frames × 960 features
    features = np.random.randn(
        SEQUENCE_LENGTH,
        FEATURE_DIM
    ).astype(np.float32)

    label_id = i % NUM_CLASSES
    label = CLASSES[label_id]

    filename = f"sample_{i:04d}.npy"

    filepath = os.path.join(
        OUTPUT_DIR,
        filename
    )

    np.save(filepath, features)

    rows.append({
        "sample_id": i,
        "feature_file": filename,
        "label_id": label_id,
        "label": label
    })


# -----------------------------
# Create labels CSV
# -----------------------------

df = pd.DataFrame(rows)

csv_path = os.path.join(
    OUTPUT_DIR,
    "labels.csv"
)

df.to_csv(csv_path, index=False)

print("Synthetic dataset created!")
print()
print(f"Number of samples : {NUM_SAMPLES}")
print(f"Sequence length   : {SEQUENCE_LENGTH}")
print(f"Feature dimension : {FEATURE_DIM}")
print(f"Number of classes : {NUM_CLASSES}")
print()
print(f"Features saved to : {OUTPUT_DIR}")
print(f"Labels saved to   : {csv_path}")