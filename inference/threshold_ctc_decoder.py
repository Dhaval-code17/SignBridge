import json
import torch


class ThresholdCTCDecoder:

    def __init__(
        self,
        classes_path,
        confidence_threshold=0.5
    ):
        with open(
            classes_path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        self.classes = [
            data["idx_to_class"][str(i)]
            for i in range(data["num_classes"])
        ]

        self.blank_id = len(self.classes)

        self.confidence_threshold = (
            confidence_threshold
        )

    def decode(self, logits):

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

        decoded_sequences = []

        for sequence_index in range(
            predictions.size(0)
        ):

            decoded_tokens = []

            previous_token = None

            for frame_index in range(
                predictions.size(1)
            ):

                token_id = predictions[
                    sequence_index,
                    frame_index
                ].item()

                confidence = probabilities[
                    sequence_index,
                    frame_index,
                    token_id
                ].item()

                # Ignore low-confidence predictions.
                if confidence < self.confidence_threshold:
                    token_id = self.blank_id

                # Ignore CTC blank.
                if token_id == self.blank_id:
                    previous_token = token_id
                    continue

                # Collapse repeated tokens.
                if token_id == previous_token:
                    continue

                if 0 <= token_id < len(self.classes):
                    decoded_tokens.append(
                        self.classes[token_id]
                    )

                previous_token = token_id

            decoded_sequences.append(
                decoded_tokens
            )

        return decoded_sequences