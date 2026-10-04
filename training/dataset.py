import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from preprocessing.video_sampling import load_uniform_frames
    from preprocessing.transforms import (
        build_eval_transform,
        build_train_transform,
        preprocess_frame,
    )
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False


class SignFeatureDataset(Dataset):
    """Dataset for pre-extracted feature vectors [T, 960]."""
    def __init__(self, csv_file, feature_dir):
        self.data = pd.read_csv(csv_file)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        feature_path = os.path.join(self.feature_dir, row["feature_file"])
        features = np.load(feature_path)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(row["label_id"], dtype=torch.long)
        return features, label


class SignVideoDataset(Dataset):
    """Dataset for raw CISLR sign videos."""
    def __init__(
        self,
        csv_path: str | Path,
        class_to_idx: dict[str, int],
        num_frames: int = 16,
        training: bool = False,
    ):
        if not PREPROCESSING_AVAILABLE:
            raise ImportError("Video preprocessing transforms could not be imported.")
        self.csv_path = Path(csv_path)
        self.class_to_idx = class_to_idx
        self.num_frames = num_frames
        self.training = training
        self.df = pd.read_csv(self.csv_path)

        self.transform = (
            build_train_transform()
            if training
            else build_eval_transform()
        )

        unknown_labels = set(self.df["gloss"]) - set(self.class_to_idx)
        if unknown_labels:
            raise ValueError(
                f"Unknown labels found in {self.csv_path}: "
                f"{sorted(unknown_labels)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        video_path = row["video_path"]
        gloss = row["gloss"]

        frames = load_uniform_frames(
            video_path,
            num_frames=self.num_frames,
        )

        tensors = [
            preprocess_frame(frame, self.transform)
            for frame in frames
        ]

        frames_tensor = torch.stack(tensors)
        label = self.class_to_idx[gloss]
        return frames_tensor, torch.tensor(label, dtype=torch.long)
