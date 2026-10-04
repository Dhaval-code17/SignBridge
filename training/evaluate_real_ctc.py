import json
import torch
from torch.utils.data import DataLoader

from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc
from models.ctc_transformer import CTCTransformer
from inference.ctc_decoder import CTCDecoder


# --------------------------------------------------
# Settings
# --------------------------------------------------

VAL_DIR = "data/continuous/val"
CLASSES_PATH = "data/person1/classes.json"
CHECKPOINT_PATH = "models/checkpoints/real_ctc_transformer.pth"

BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load classes
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
# Decoder
# --------------------------------------------------

decoder = CTCDecoder(
    CLASSES_PATH
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

total_sequences = 0
exact_matches = 0

total_true_signs = 0
total_correct_signs = 0

examples_shown = 0
MAX_EXAMPLES = 20


with torch.no_grad():

    for (
        features,
        feature_lengths,
        targets,
        target_lengths
    ) in loader:

        features = features.to(DEVICE)

        logits = model(features)

        decoded_sequences = decoder.decode(logits)

        # ------------------------------------------
        # Recover true labels for each sequence
        # ------------------------------------------

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

            predicted_labels = decoded_sequences[
                batch_index
            ]

            # --------------------------------------
            # Exact sequence match
            # --------------------------------------

            if predicted_labels == true_labels:
                exact_matches += 1

            total_sequences += 1

            # --------------------------------------
            # Sign-level accuracy
            #
            # Compare positions that exist in both
            # sequences.
            # --------------------------------------

            compare_length = min(
                len(true_labels),
                len(predicted_labels)
            )

            for i in range(compare_length):

                if (
                    true_labels[i]
                    == predicted_labels[i]
                ):
                    total_correct_signs += 1

            total_true_signs += len(true_labels)

            # --------------------------------------
            # Print examples
            # --------------------------------------

            if examples_shown < MAX_EXAMPLES:

                print()
                print(
                    f"Example {examples_shown + 1}"
                )

                print(
                    "True:     ",
                    " | ".join(true_labels)
                )

                print(
                    "Predicted:",
                    " | ".join(predicted_labels)
                )

                examples_shown += 1


# --------------------------------------------------
# Metrics
# --------------------------------------------------

sequence_accuracy = (
    exact_matches / total_sequences
    if total_sequences > 0
    else 0.0
)

sign_accuracy = (
    total_correct_signs / total_true_signs
    if total_true_signs > 0
    else 0.0
)


print()
print("======================================")
print("Evaluation Results")
print("======================================")

print(
    f"Validation sequences: {total_sequences}"
)

print(
    f"Exact sequence matches: {exact_matches}"
)

print(
    f"Sequence accuracy: "
    f"{sequence_accuracy * 100:.2f}%"
)

print(
    f"Correct signs: {total_correct_signs}"
)

print(
    f"Total true signs: {total_true_signs}"
)

print(
    f"Position-wise sign accuracy: "
    f"{sign_accuracy * 100:.2f}%"
)