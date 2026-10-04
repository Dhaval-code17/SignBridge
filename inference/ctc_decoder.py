import json
import torch


class CTCDecoder:

    def __init__(self, classes_path):

        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Your classes.json format:
        #
        # {
        #     "num_classes": 39,
        #     "class_to_idx": {...},
        #     "idx_to_class": {...}
        # }

        if not isinstance(data, dict):
            raise ValueError(
                "Expected classes.json to contain a dictionary."
            )

        if "idx_to_class" not in data:
            raise ValueError(
                "classes.json does not contain 'idx_to_class'."
            )

        id_to_class = data["idx_to_class"]

        num_classes = data.get(
            "num_classes",
            len(id_to_class)
        )

        self.classes = [
            id_to_class[str(i)]
            for i in range(num_classes)
        ]

        # CTC blank is after all real classes.
        # 39 real classes -> blank ID = 39
        self.blank_id = len(self.classes)

        print("Loaded classes:", len(self.classes))
        print("First classes:", self.classes[:10])
        print("Blank ID:", self.blank_id)

    def decode(self, logits):

        # logits shape:
        # [batch, time, 40]

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        decoded_sequences = []

        for sequence in predictions:

            decoded_tokens = []
            previous_token = None

            for token_id in sequence.tolist():

                # Ignore CTC blank
                if token_id == self.blank_id:
                    previous_token = token_id
                    continue

                # Remove repeated consecutive tokens
                if token_id == previous_token:
                    continue

                # Convert ID -> class name
                if 0 <= token_id < len(self.classes):
                    decoded_tokens.append(
                        self.classes[token_id]
                    )

                previous_token = token_id

            decoded_sequences.append(
                decoded_tokens
            )

        return decoded_sequences


if __name__ == "__main__":

    classes_path = "data/person1/classes.json"

    decoder = CTCDecoder(classes_path)

    num_classes = len(decoder.classes)

    # 39 signs + 1 CTC blank = 40 outputs
    logits = torch.zeros(
        1,
        10,
        num_classes + 1
    )

    # Find class IDs
    monday_id = decoder.classes.index("monday")
    india_id = decoder.classes.index("india")
    thank_you_id = decoder.classes.index("thank you")

    blank_id = decoder.blank_id

    # Fake CTC prediction:
    #
    # monday monday blank
    # india india blank
    # thank you thank you blank blank

    token_sequence = [
        monday_id,
        monday_id,
        blank_id,
        india_id,
        india_id,
        blank_id,
        thank_you_id,
        thank_you_id,
        blank_id,
        blank_id
    ]

    for t, token_id in enumerate(token_sequence):
        logits[0, t, token_id] = 10.0

    decoded = decoder.decode(logits)

    print("\nDecoded sequence:")
    print(decoded)