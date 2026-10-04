import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SignSegmentDataset(Dataset):
    def __init__(self, data_dir, class_to_idx):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx

        self.samples = []

        # Each sequence has:
        #   seq_XXXXXX.npy
        #   seq_XXXXXX.json
        #
        # The JSON contains the sign labels.
        #
        # The metadata tells us the segment lengths, so we need
        # metadata.json to recover the individual sign boundaries.

        metadata_path = os.path.join(
            data_dir,
            "metadata.json"
        )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found in {data_dir}"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for sequence in metadata["sequences"]:

            sequence_id = sequence["sequence_id"]

            feature_path = os.path.join(
                data_dir,
                f"{sequence_id}.npy"
            )

            if not os.path.exists(feature_path):
                continue

            labels = sequence["labels"]
            segment_lengths = sequence["segment_lengths"]

            if len(labels) != len(segment_lengths):
                raise ValueError(
                    f"{sequence_id}: number of labels does not "
                    f"match number of segment lengths"
                )

            start = 0

            for label, segment_length in zip(
                labels,
                segment_lengths
            ):

                end = start + segment_length

                self.samples.append({
                    "feature_path": feature_path,
                    "start": start,
                    "end": end,
                    "label": class_to_idx[label],
                    "label_name": label,
                })

                start = end

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        features = np.load(
            sample["feature_path"]
        )

        segment = features[
            sample["start"]:sample["end"]
        ]

        segment = torch.tensor(
            segment,
            dtype=torch.float32
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return segment, label