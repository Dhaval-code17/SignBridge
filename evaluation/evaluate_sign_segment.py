import json
import torch
from collections import Counter
from torch.utils.data import DataLoader

from training.sign_segment_dataset import SignSegmentDataset
from models.sign_segment_classifier import SignSegmentClassifier


DATA_DIR = "data/continuous/train"
CLASSES_PATH = "data/person1/classes.json"
CHECKPOINT_PATH = "models/checkpoints/sign_segment_classifier.pth"

BATCH_SIZE = 8


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
    print("Individual Sign Segment Evaluation")
    print("=" * 50)

    device = torch.device("cpu")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_data = json.load(f)

    class_to_idx = classes_data["class_to_idx"]
    idx_to_class = {
        int(k): v
        for k, v in classes_data["idx_to_class"].items()
    }

    dataset = SignSegmentDataset(
        DATA_DIR,
        class_to_idx,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_segments,
    )

    model = SignSegmentClassifier(
        input_dim=960,
        hidden_dim=256,
        num_layers=2,
        num_classes=39,
        dropout=0.3,
    )

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    total = 0
    correct = 0

    true_counts = Counter()
    correct_counts = Counter()
    confusion = Counter()

    with torch.no_grad():

        for features, lengths, labels in loader:

            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(
                features,
                lengths,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            for true_id, predicted_id in zip(
                labels.tolist(),
                predictions.tolist(),
            ):

                total += 1
                true_counts[true_id] += 1

                if true_id == predicted_id:
                    correct += 1
                    correct_counts[true_id] += 1
                else:
                    confusion[
                        (true_id, predicted_id)
                    ] += 1

    accuracy = 100.0 * correct / total

    print()
    print("Total segments:", total)
    print("Correct:", correct)
    print(f"Accuracy: {accuracy:.2f}%")

    print()
    print("Per-sign accuracy:")
    print("-" * 55)

    for class_id in sorted(true_counts):

        count = true_counts[class_id]
        correct_count = correct_counts[class_id]

        sign_accuracy = (
            100.0 * correct_count / count
        )

        print(
            f"{idx_to_class[class_id]:20s} "
            f"{correct_count:3d}/{count:3d} "
            f"({sign_accuracy:6.2f}%)"
        )

    print()
    print("Confusions:")
    print("-" * 55)

    for (true_id, predicted_id), count in confusion.most_common(20):

        print(
            f"{idx_to_class[true_id]:20s} -> "
            f"{idx_to_class[predicted_id]:20s} : "
            f"{count}"
        )


if __name__ == "__main__":
    main()