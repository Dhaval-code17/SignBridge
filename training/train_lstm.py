import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.dataset import SignFeatureDataset
from models.lstm import LSTMClassifier


# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_CLASSES = 39


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Dataset
# -----------------------------

dataset = SignFeatureDataset(
    csv_file="data/synthetic/labels.csv",
    feature_dir="data/synthetic"
)

print("Total samples:", len(dataset))


# -----------------------------
# Train / validation / test split
# -----------------------------

total_size = len(dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# -----------------------------
# DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------

model = LSTMClassifier(
    input_dim=960,
    hidden_dim=256,
    num_layers=2,
    num_classes=NUM_CLASSES
)

model = model.to(device)


# -----------------------------
# Loss and optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_accuracy = 100 * correct / total

    average_loss = (
        total_loss / len(train_loader)
    )


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# -----------------------------
# Save model
# -----------------------------

import os

os.makedirs("models/checkpoints", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/checkpoints/lstm_baseline.pth"
)

print()
print("LSTM model saved!")