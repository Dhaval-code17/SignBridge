import json
import torch

from models.ctc_transformer import CTCTransformer
from inference.ctc_decoder import CTCDecoder
from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc

from torch.utils.data import DataLoader


# --------------------------------------------------
# Settings
# --------------------------------------------------

VAL_DIR = "data/continuous/val"
CLASSES_PATH = "data/person1/classes.json"
CHECKPOINT_PATH = "models/checkpoints/real_ctc_transformer.pth"

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
idx_to_class = class_data["idx_to_class"]

NUM_CLASSES = len(class_to_idx)
BLANK_ID = NUM_CLASSES


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = RealCTCDataset(
    data_dir=VAL_DIR,
    class_to_idx=class_to_idx
)

loader = DataLoader(
    dataset,
    batch_size=1,
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
# Debug first 5 validation sequences
# --------------------------------------------------

with torch.no_grad():

    for example_number, batch in enumerate(loader):

        if example_number >= 5:
            break

        features, feature_lengths, targets, target_lengths = batch

        features = features.to(DEVICE)

        logits = model(features)

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

        sequence_length = feature_lengths[0].item()

        predictions = predictions[0][:sequence_length]
        probabilities = probabilities[0][:sequence_length]

        # ------------------------------------------
        # True labels
        # ------------------------------------------

        target_length = target_lengths[0].item()

        true_ids = targets[0:target_length]

        true_labels = [
            idx_to_class[str(x.item())]
            for x in true_ids
        ]

        # ------------------------------------------
        # Print
        # ------------------------------------------

        print()
        print("======================================")
        print(
            f"Example {example_number + 1}"
        )
        print("======================================")

        print(
            "True sequence:",
            " | ".join(true_labels)
        )

        print()
        print("Frame predictions:")

        previous_prediction = None

        for frame_index in range(sequence_length):

            prediction_id = predictions[
                frame_index
            ].item()

            probability = probabilities[
                frame_index,
                prediction_id
            ].item()

            if prediction_id == BLANK_ID:
                label = "<BLANK>"
            else:
                label = idx_to_class[
                    str(prediction_id)
                ]

            # Only print when prediction changes.
            if prediction_id != previous_prediction:

                print(
                    f"Frame {frame_index:3d}: "
                    f"{label:15s} "
                    f"confidence={probability:.3f}"
                )

            previous_prediction = prediction_id