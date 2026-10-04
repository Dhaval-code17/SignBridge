from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.video_sampling import load_uniform_frames
from preprocessing.transforms import (
    build_eval_transform,
    build_train_transform,
    preprocess_frame,
)


class SignVideoDataset(Dataset):
    """
    Dataset for CISLR sign videos.

    Returns:
        frames: Tensor [T, C, H, W]
        label: Integer class ID
    """

    def __init__(
        self,
        csv_path: str | Path,
        class_to_idx: dict[str, int],
        num_frames: int = 16,
        training: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.class_to_idx = class_to_idx
        self.num_frames = num_frames
        self.training = training

        self.df = pd.read_csv(self.csv_path)

        # Select augmentation pipeline once.
        self.transform = (
            build_train_transform()
            if training
            else build_eval_transform()
        )

        # Validate labels.
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