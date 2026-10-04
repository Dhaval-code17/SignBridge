import json
import torch
from torch.utils.data import DataLoader

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc
from models.ctc_transformer import CTCTransformer
from inference.threshold_ctc_decoder import ThresholdCTCDecoder
from evaluation.ctc_metrics import edit_distance


VAL_DIR = "data/continuous/val"
CLASSES_PATH = "data/person1/classes.json"
CHECKPOINT_PATH = "models/checkpoints/real_ctc_transformer.pth"

BATCH_SIZE = 4

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Classes
# --------------------------------------------------

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

NUM_CLASSES = len(class_to_idx)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = RealCTCDataset(
    data_dir=VAL_DIR,
    class_to_idx=class_to_idx
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_real_ctc
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = CTCTransformer(
    input_dim=960,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    num_classes=NUM_CLASSES,
    dropout=0.1
)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Evaluate each threshold
# --------------------------------------------------

print()
print("==============================================")
print("CTC Decoder Threshold Comparison")
print("==============================================")


for threshold in THRESHOLDS:

    decoder = ThresholdCTCDecoder(
        CLASSES_PATH,
        confidence_threshold=threshold
    )

    total_sequences = 0
    exact_matches = 0

    total_reference_signs = 0
    total_edit_distance = 0

    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0

    with torch.no_grad():

        for (
            features,
            feature_lengths,
            targets,
            target_lengths
        ) in loader:

            features = features.to(DEVICE)

            logits = model(features)

            predictions = decoder.decode(
                logits
            )

            target_offset = 0

            for batch_index in range(
                len(target_lengths)
            ):

                target_length = target_lengths[
                    batch_index
                ].item()

                target_ids = targets[
                    target_offset:
                    target_offset + target_length
                ]

                target_offset += target_length

                true_labels = [
                    class_data["idx_to_class"][
                        str(target_id.item())
                    ]
                    for target_id in target_ids
                ]

                predicted_labels = predictions[
                    batch_index
                ]

                (
                    distance,
                    substitutions,
                    insertions,
                    deletions
                ) = edit_distance(
                    true_labels,
                    predicted_labels
                )

                total_edit_distance += distance

                total_substitutions += substitutions
                total_insertions += insertions
                total_deletions += deletions

                total_reference_signs += len(
                    true_labels
                )

                total_sequences += 1

                if true_labels == predicted_labels:
                    exact_matches += 1

    normalized_edit_distance = (
        total_edit_distance
        / total_reference_signs
    )

    sequence_accuracy = (
        exact_matches
        / total_sequences
    )

    print()
    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"  Edit distance: "
        f"{total_edit_distance}"
    )

    print(
        f"  Normalized edit distance: "
        f"{normalized_edit_distance:.4f}"
    )

    print(
        f"  Substitutions: "
        f"{total_substitutions}"
    )

    print(
        f"  Insertions: "
        f"{total_insertions}"
    )

    print(
        f"  Deletions: "
        f"{total_deletions}"
    )

    print(
        f"  Exact sequence accuracy: "
        f"{sequence_accuracy * 100:.2f}%"
    )