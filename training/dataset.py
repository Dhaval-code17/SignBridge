import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SignFeatureDataset(Dataset):

    def __init__(self, csv_file, feature_dir):

        self.data = pd.read_csv(csv_file)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        feature_path = os.path.join(
            self.feature_dir,
            row["feature_file"]
        )

        # Load [16, 960]
        features = np.load(feature_path)

        # Convert to PyTorch tensor
        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        # Class ID
        label = torch.tensor(
            row["label_id"],
            dtype=torch.long
        )

        return features, label


if __name__ == "__main__":

    dataset = SignFeatureDataset(
        csv_file="data/synthetic/labels.csv",
        feature_dir="data/synthetic"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    features, labels = next(iter(dataloader))

    print("Batch feature shape:", features.shape)
    print("Batch label shape:", labels.shape)
    print("First labels:", labels[:10])