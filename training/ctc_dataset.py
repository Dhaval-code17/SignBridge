import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CTCSyntheticDataset(Dataset):

    def __init__(
        self,
        num_samples=200,
        min_length=20,
        max_length=40,
        feature_dim=960,
        num_classes=39,
        max_target_length=5,
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.max_target_length = max_target_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        # Random sequence length
        sequence_length = np.random.randint(
            self.min_length,
            self.max_length + 1
        )

        # Simulated Person 1 features
        features = np.random.randn(
            sequence_length,
            self.feature_dim
        ).astype(np.float32)

        # Random target sequence
        target_length = np.random.randint(
            1,
            self.max_target_length + 1
        )

        targets = np.random.randint(
            0,
            self.num_classes,
            size=target_length
        )

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        targets = torch.tensor(
            targets,
            dtype=torch.long
        )

        return features, targets