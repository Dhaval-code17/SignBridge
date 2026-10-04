import torch
from torch.utils.data import DataLoader

from training.sequence_dataset import (
    ContinuousFeatureDataset
)

from training.collate import (
    collate_sequences
)


# --------------------------------
# Person 1 real feature sequence
# --------------------------------

feature_files = [
    "data/person1/continuous_feature_sequence.npy"
]


# --------------------------------
# Create dataset
# --------------------------------

dataset = ContinuousFeatureDataset(
    feature_files=feature_files
)


# --------------------------------
# Create DataLoader
# --------------------------------

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_sequences
)


# --------------------------------
# Get one batch
# --------------------------------

features, padding_mask = next(
    iter(loader)
)


print("Features shape:")
print(features.shape)

print()

print("Padding mask shape:")
print(padding_mask.shape)

print()

print("Padding mask:")
print(padding_mask)