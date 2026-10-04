import torch
from torch.nn.utils.rnn import pad_sequence


def collate_ctc(batch):

    features, targets = zip(*batch)

    # ------------------------------------------------
    # 1. Pad feature sequences
    # ------------------------------------------------
    #
    # Example:
    #
    # [25, 960]
    # [33, 960]
    # [28, 960]
    #
    # becomes:
    #
    # [3, 33, 960]
    #
    # The extra frames are zero-padded.

    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )

    # ------------------------------------------------
    # 2. Create feature lengths
    # ------------------------------------------------

    feature_lengths = torch.tensor(
        [feature.size(0) for feature in features],
        dtype=torch.long
    )

    # ------------------------------------------------
    # 3. Keep target sequences unpadded
    # ------------------------------------------------
    #
    # CTC loss accepts targets as one concatenated
    # tensor plus target lengths.

    target_lengths = torch.tensor(
        [target.size(0) for target in targets],
        dtype=torch.long
    )

    concatenated_targets = torch.cat(
        targets
    )

    return (
        padded_features,
        feature_lengths,
        concatenated_targets,
        target_lengths
    )