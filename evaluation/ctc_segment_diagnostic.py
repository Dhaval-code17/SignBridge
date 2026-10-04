import json
import torch
from torch.utils.data import DataLoader

from models.ctc_transformer import CTCTransformer
from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc


CHECKPOINT_PATH = "models/checkpoints/real_ctc_transformer.pth"
CLASSES_PATH = "data/person1/classes.json"
VAL_DIR = "data/continuous/val"

BATCH_SIZE = 1


def main():

    print("=" * 60)
    print("CTC Segment Probability Diagnostic")
    print("=" * 60)

    device = torch.device("cpu")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_data = json.load(f)

    class_to_idx = classes_data["class_to_idx"]

    idx_to_class = {
        int(k): v
        for k, v in classes_data["idx_to_class"].items()
    }

    dataset = RealCTCDataset(
        VAL_DIR,
        class_to_idx,
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

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    with open(
        f"{VAL_DIR}/metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    metadata_by_id = {
        item["sequence_id"]: item
        for item in metadata["sequences"]
    }

    sequence_count = 0

    with torch.no_grad():

        for batch_index, (
            features,
            feature_lengths,
            targets,
            target_lengths,
        ) in enumerate(loader):

            if sequence_count >= 10:
                break

            features = features.to(device)

            logits = model(features)

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

            feature_file = dataset.feature_files[batch_index]

            sequence_id = feature_file.replace(
                ".npy",
                "",
            )

            metadata_item = metadata_by_id[sequence_id]

            labels = metadata_item["labels"]
            segment_lengths = metadata_item["segment_lengths"]

            print()
            print("=" * 60)
            print(sequence_id)
            print("=" * 60)

            frame_start = 0

            for label, segment_length in zip(
                labels,
                segment_lengths,
            ):

                frame_end = frame_start + segment_length

                true_id = class_to_idx[label]

                segment_probs = probabilities[
                    frame_start:frame_end
                ]

                true_sign_probs = segment_probs[
                    :, true_id
                ]

                max_probability = (
                    true_sign_probs.max().item()
                )

                mean_probability = (
                    true_sign_probs.mean().item()
                )

                max_frame = (
                    true_sign_probs.argmax().item()
                    + frame_start
                )

                max_frame_prediction = probabilities[
                    max_frame
                ].argmax().item()

                predicted_name = (
                    "blank"
                    if max_frame_prediction == 39
                    else idx_to_class.get(
                        max_frame_prediction,
                        "unknown",
                    )
                )

                print()
                print(
                    f"Frames {frame_start}-{frame_end - 1}"
                )
                print(
                    f"True sign: {label}"
                )
                print(
                    f"Maximum true-sign probability: "
                    f"{max_probability:.4f}"
                )
                print(
                    f"Mean true-sign probability: "
                    f"{mean_probability:.4f}"
                )
                print(
                    f"Best frame: {max_frame}"
                )
                print(
                    f"Prediction at best frame: "
                    f"{predicted_name}"
                )

                # Show the strongest predictions in this segment.
                segment_max_probs, segment_ids = (
                    segment_probs.max(dim=0)
                )

                top_values, top_ids = torch.topk(
                    segment_max_probs,
                    k=5,
                )

                print("Top signs by maximum probability:")

                for value, token_id in zip(
                    top_values.tolist(),
                    top_ids.tolist(),
                ):
                    name = (
                        "blank"
                        if token_id == 39
                        else idx_to_class[token_id]
                    )

                    print(
                        f"  {name:15s} "
                        f"{value:.4f}"
                    )

                frame_start = frame_end

            sequence_count += 1


if __name__ == "__main__":
    main()