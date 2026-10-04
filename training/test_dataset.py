import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import SignVideoDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_classes():
    path = PROJECT_ROOT / "configs" / "classes.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["class_to_idx"]


def main():
    class_to_idx = load_classes()

    train_dataset = SignVideoDataset(
        csv_path=PROJECT_ROOT / "data" / "processed" / "train.csv",
        class_to_idx=class_to_idx,
        num_frames=16,
        training=True,
    )

    val_dataset = SignVideoDataset(
        csv_path=PROJECT_ROOT / "data" / "processed" / "val.csv",
        class_to_idx=class_to_idx,
        num_frames=16,
        training=False,
    )

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    frames, labels = next(iter(train_loader))

    print("\nTrain batch:")
    print("Frames:", frames.shape)
    print("Labels:", labels.shape)
    print("Frames dtype:", frames.dtype)
    print("Labels dtype:", labels.dtype)
    print("Labels:", labels)

    frames, labels = next(iter(val_loader))

    print("\nValidation batch:")
    print("Frames:", frames.shape)
    print("Labels:", labels.shape)

    # Basic sanity checks.
    assert frames.ndim == 5
    assert frames.shape[1:] == (16, 3, 224, 224)
    assert labels.ndim == 1
    assert frames.dtype == torch.float32
    assert labels.dtype == torch.long

    print("\nDataset/DataLoader test successful.")


if __name__ == "__main__":
    main()