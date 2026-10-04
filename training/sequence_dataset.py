import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ContinuousFeatureDataset(Dataset):

    def __init__(self, feature_files, labels=None):
        self.feature_files = feature_files
        self.labels = labels

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, index):

        feature_path = self.feature_files[index]

        features = np.load(feature_path)

        # Expected shape:
        # [T, 960]
        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        if self.labels is not None:
            label = torch.tensor(
                self.labels[index],
                dtype=torch.long
            )

            return features, label

        return features