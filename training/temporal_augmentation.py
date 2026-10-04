import random
import torch


def temporal_augment(
    features,
    drop_probability=0.10,
    duplicate_probability=0.10,
):
    """
    Apply small temporal changes to a [T, 960] feature sequence.

    The feature values themselves are not changed.
    Only the temporal sampling is changed.
    """

    if features.size(0) < 3:
        return features

    frames = []

    for i in range(features.size(0)):

        # Randomly drop a frame
        if (
            random.random() < drop_probability
            and len(frames) > 1
        ):
            continue

        frames.append(features[i])

        # Randomly duplicate a frame
        if random.random() < duplicate_probability:
            frames.append(features[i])

    # Make sure at least 2 frames remain
    if len(frames) < 2:
        return features

    return torch.stack(frames)