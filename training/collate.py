import torch
from torch.nn.utils.rnn import pad_sequence


def collate_sequences(batch):

    # If labels are present
    if isinstance(batch[0], tuple):

        sequences, labels = zip(*batch)

        labels = torch.stack(labels)

    else:

        sequences = batch
        labels = None

    # Find maximum sequence length
    max_length = max(
        sequence.size(0)
        for sequence in sequences
    )

    # Pad sequences
    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    # Create padding mask
    #
    # False = real data
    # True  = padding
    padding_mask = torch.ones(
        len(sequences),
        max_length,
        dtype=torch.bool
    )

    for i, sequence in enumerate(sequences):

        length = sequence.size(0)

        padding_mask[
            i,
            :length
        ] = False

    if labels is not None:

        return padded, padding_mask, labels

    return padded, padding_mask