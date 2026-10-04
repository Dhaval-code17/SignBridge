import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class RealCTCDataset(Dataset):
    def __init__(self, data_dir, class_to_idx):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx

        self.feature_files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.endswith(".npy")
            ]
        )

        if len(self.feature_files) == 0:
            raise ValueError(
                f"No .npy files found in {data_dir}"
            )

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, index):
        feature_file = self.feature_files[index]

        feature_path = os.path.join(
            self.data_dir,
            feature_file
        )

        label_file = feature_file.replace(
            ".npy",
            ".json"
        )

        label_path = os.path.join(
            self.data_dir,
            label_file
        )

        features = np.load(feature_path)

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:
            labels = json.load(f)

        target = torch.tensor(
            [
                self.class_to_idx[label]
                for label in labels
            ],
            dtype=torch.long
        )

        return features, target