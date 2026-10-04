import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.sign_segment_dataset import SignSegmentDataset
from models.sign_segment_classifier import SignSegmentClassifier


DATA_DIR = "data/continuous/train"
CLASSES_PATH = "data/person1/classes.json"
CHECKPOINT_PATH = "models/checkpoints/sign_segment_classifier.pth"

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0005


def pad_segments(batch):
    segments, labels = zip(*batch)

    lengths = torch.tensor(
        [segment.size(0) for segment in segments],
        dtype=torch.long,
    )

    max_length = max(
        segment.size(0)
        for segment in segments
    )

    padded = torch.zeros(
        len(segments),
        max_length,
        segments[0].size(1),
        dtype=torch.float32,
    )

    for i, segment in enumerate(segments):
        padded[i, :segment.size(0)] = segment

    labels = torch.stack(labels)

    return padded, lengths, labels


def main():

    print("=" * 50)
    print("Individual Sign Segment Classifier")
    print("=" * 50)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_data = json.load(f)

    class_to_idx = classes_data["class_to_idx"]
    num_classes = classes_data["num_classes"]

    dataset = SignSegmentDataset(
        DATA_DIR,
        class_to_idx,
    )

    print("Total sign segments:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print("Training segments:", len(train_dataset))
    print("Validation segments:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_segments,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_segments,
    )

    model = SignSegmentClassifier(
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, lengths, labels in train_loader:

            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features, lengths)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        # Validation
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for features, lengths, labels in val_loader:

                features = features.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                logits = model(features, lengths)

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = 100.0 * val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_accuracy:.2f}% "
            f"- Val Acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH,
            )

    print()
    print("Training complete!")
    print("Best validation accuracy:", f"{best_val_accuracy:.2f}%")
    print("Model saved to:")
    print(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()