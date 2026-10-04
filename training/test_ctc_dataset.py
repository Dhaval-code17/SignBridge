from torch.utils.data import DataLoader

from training.ctc_dataset import CTCSyntheticDataset
from training.ctc_collate import collate_ctc


dataset = CTCSyntheticDataset(
    num_samples=10
)

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=collate_ctc
)


for (
    features,
    feature_lengths,
    targets,
    target_lengths
) in loader:

    print("Features shape:", features.shape)
    print("Feature lengths:", feature_lengths)

    print("Targets shape:", targets.shape)
    print("Targets:", targets)

    print("Target lengths:", target_lengths)

    break