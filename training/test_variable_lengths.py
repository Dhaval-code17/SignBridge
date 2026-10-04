import torch
from torch.utils.data import DataLoader
from training.sequence_dataset import ContinuousFeatureDataset
from training.collate import collate_sequences


# --------------------------------
# Temporary sequences
# --------------------------------

sequence_1 = torch.randn(24, 960)
sequence_2 = torch.randn(31, 960)
sequence_3 = torch.randn(18, 960)


# Save temporary files
torch.save(sequence_1, "data/person1/temp_24.pt")
torch.save(sequence_2, "data/person1/temp_31.pt")
torch.save(sequence_3, "data/person1/temp_18.pt")


# For this test, create dataset directly
class TemporaryDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.data = [
            sequence_1,
            sequence_2,
            sequence_3
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


dataset = TemporaryDataset()


loader = DataLoader(
    dataset,
    batch_size=3,
    collate_fn=collate_sequences
)


features, padding_mask = next(
    iter(loader)
)


print("Batch shape:")
print(features.shape)

print()

print("Padding mask:")
print(padding_mask)