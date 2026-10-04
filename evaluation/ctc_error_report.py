import json
import torch
from collections import Counter
from torch.utils.data import DataLoader

from models.ctc_transformer import CTCTransformer
from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc
from inference.threshold_ctc_decoder import ThresholdCTCDecoder


CHECKPOINT_PATH = "models/checkpoints/real_ctc_transformer.pth"
CLASSES_PATH = "data/person1/classes.json"
VAL_DIR = "data/continuous/val"

THRESHOLD = 0.70
BATCH_SIZE = 4


def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["class_to_idx"]


def align_sequences(true_sequence, predicted_sequence):
    """
    Align two sequences using simple dynamic programming.

    Returns:
        substitutions: list of (true, predicted)
        insertions: list of predicted signs
        deletions: list of true signs
    """

    n = len(true_sequence)
    m = len(predicted_sequence)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            substitution_cost = (
                0
                if true_sequence[i - 1] == predicted_sequence[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    substitutions = []
    insertions = []
    deletions = []

    i = n
    j = m

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and true_sequence[i - 1] == predicted_sequence[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1

        elif (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            substitutions.append(
                (true_sequence[i - 1], predicted_sequence[j - 1])
            )
            i -= 1
            j -= 1

        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions.append(true_sequence[i - 1])
            i -= 1

        else:
            insertions.append(predicted_sequence[j - 1])
            j -= 1

    substitutions.reverse()
    insertions.reverse()
    deletions.reverse()

    return substitutions, insertions, deletions


def main():

    print("=" * 50)
    print("CTC Error Report")
    print("=" * 50)

    device = torch.device("cpu")

    class_to_idx = load_classes()

    dataset = RealCTCDataset(
        data_dir=VAL_DIR,
        class_to_idx=class_to_idx,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
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

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    decoder = ThresholdCTCDecoder(
        CLASSES_PATH,
        confidence_threshold=THRESHOLD,
    )

    substitution_counter = Counter()
    insertion_counter = Counter()
    deletion_counter = Counter()

    with torch.no_grad():

        for features, feature_lengths, targets, target_lengths in loader:

            features = features.to(device)

            logits = model(features)

            predictions = decoder.decode(logits)

            target_offset = 0

            for batch_index in range(len(feature_lengths)):

                target_length = target_lengths[batch_index].item()

                target_ids = targets[
                    target_offset:
                    target_offset + target_length
                ]

                target_offset += target_length

                true_sequence = [
                    next(
                        name
                        for name, idx in class_to_idx.items()
                        if idx == token_id.item()
                    )
                    for token_id in target_ids
                ]

                predicted_sequence = predictions[batch_index]

                substitutions, insertions, deletions = align_sequences(
                    true_sequence,
                    predicted_sequence,
                )

                substitution_counter.update(substitutions)
                insertion_counter.update(insertions)
                deletion_counter.update(deletions)

    print()
    print("Threshold:", THRESHOLD)

    print()
    print("Top substitutions:")
    print("-" * 50)

    for (true_sign, predicted_sign), count in substitution_counter.most_common(20):
        print(
            f"{true_sign:20s} -> {predicted_sign:20s} : {count}"
        )

    print()
    print("Top insertions:")
    print("-" * 50)

    for sign, count in insertion_counter.most_common(20):
        print(
            f"{sign:20s} : {count}"
        )

    print()
    print("Top deletions:")
    print("-" * 50)

    for sign, count in deletion_counter.most_common(20):
        print(
            f"{sign:20s} : {count}"
        )


if __name__ == "__main__":
    main()