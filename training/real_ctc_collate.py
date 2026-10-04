import torch
from torch.nn.utils.rnn import pad_sequence


def collate_real_ctc(batch):
    features, targets = zip(*batch)

    # Pad feature sequences to the longest sequence
    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )

    # Original length of every feature sequence
    feature_lengths = torch.tensor(
        [x.size(0) for x in features],
        dtype=torch.long
    )

    # Length of every target sequence
    target_lengths = torch.tensor(
        [x.size(0) for x in targets],
        dtype=torch.long
    )

    # CTC expects all targets concatenated into one tensor
    concatenated_targets = torch.cat(targets)

    return (
        padded_features,
        feature_lengths,
        concatenated_targets,
        target_lengths
    )