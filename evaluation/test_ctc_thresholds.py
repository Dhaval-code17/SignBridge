import json
import torch
from torch.utils.data import DataLoader

from models.ctc_transformer import CTCTransformer
from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc


CHECKPOINT_PATH = "models/checkpoints/real_ctc_transformer.pth"
CLASSES_PATH = "data/person1/classes.json"
VAL_DIR = "data/continuous/val"

BLANK_ID = 39


def edit_distance(reference, hypothesis):
    m = len(reference)
    n = len(hypothesis)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):

            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]


def decode(logits, threshold):

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1,
    )

    results = []

    for sequence_index in range(predictions.size(0)):

        decoded = []
        previous_token = BLANK_ID

        for frame_index in range(predictions.size(1)):

            token = predictions[
                sequence_index,
                frame_index,
            ].item()

            confidence = probabilities[
                sequence_index,
                frame_index,
                token,
            ].item()

            if confidence < threshold:
                token = BLANK_ID

            if token == BLANK_ID:
                previous_token = BLANK_ID
                continue

            if token == previous_token:
                continue

            decoded.append(token)
            previous_token = token

        results.append(decoded)

    return results


def main():

    print("=" * 60)
    print("CTC Threshold Evaluation")
    print("=" * 60)

    device = torch.device("cpu")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    class_to_idx = classes["class_to_idx"]

    dataset = RealCTCDataset(
        VAL_DIR,
        class_to_idx,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_real_ctc,
    )

    model = CTCTransformer(
        input_dim=960,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=39,
        dropout=0.1,
    )

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = {}

    with torch.no_grad():

        for threshold in thresholds:

            total_edit_distance = 0
            total_reference_signs = 0
            exact_matches = 0
            total_sequences = 0

            for (
                features,
                feature_lengths,
                targets,
                target_lengths,
            ) in loader:

                features = features.to(device)

                logits = model(features)

                predictions = decode(
                    logits,
                    threshold,
                )

                target_offset = 0

                for i in range(len(predictions)):

                    target_length = (
                        target_lengths[i].item()
                    )

                    reference = targets[
                        target_offset:
                        target_offset + target_length
                    ].tolist()

                    target_offset += target_length

                    hypothesis = predictions[i]

                    distance = edit_distance(
                        reference,
                        hypothesis,
                    )

                    total_edit_distance += distance
                    total_reference_signs += len(reference)

                    if reference == hypothesis:
                        exact_matches += 1

                    total_sequences += 1

            normalized = (
                total_edit_distance
                / total_reference_signs
            )

            exact_accuracy = (
                100.0
                * exact_matches
                / total_sequences
            )

            results[threshold] = (
                normalized,
                exact_accuracy,
                total_edit_distance,
            )

            print(
                f"Threshold: {threshold:.2f} | "
                f"Edit Distance: {total_edit_distance} | "
                f"Normalized: {normalized:.4f} | "
                f"Exact Accuracy: {exact_accuracy:.2f}%"
            )

    print()
    print("=" * 60)
    print("Best threshold by normalized edit distance")
    print("=" * 60)

    best = min(
        results.items(),
        key=lambda x: x[1][0],
    )

    print(
        f"Threshold: {best[0]:.2f}"
    )

    print(
        f"Normalized edit distance: "
        f"{best[1][0]:.4f}"
    )

    print(
        f"Exact sequence accuracy: "
        f"{best[1][1]:.2f}%"
    )


if __name__ == "__main__":
    main()