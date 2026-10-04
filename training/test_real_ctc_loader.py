import json
from torch.utils.data import DataLoader

from real_ctc_dataset import RealCTCDataset
from real_ctc_collate import collate_real_ctc


# Load Person 1's class mapping
with open(
    "data/person1/classes.json",
    "r",
    encoding="utf-8"
) as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]


# Create dataset
dataset = RealCTCDataset(
    data_dir="data/continuous",
    class_to_idx=class_to_idx
)

print("Dataset size:", len(dataset))


# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_real_ctc
)


# Get one batch
features, feature_lengths, targets, target_lengths = next(iter(loader))


print("\nBatch test:")
print("Features shape:", features.shape)
print("Feature lengths:", feature_lengths)
print("Targets:", targets)
print("Target lengths:", target_lengths)
print("Total targets:", targets.numel())