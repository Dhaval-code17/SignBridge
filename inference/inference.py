from pathlib import Path

import numpy as np

from training.extract_continuous_features import (
    load_model,
    extract_continuous_features,
)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

_MODEL = None


def _get_model():

    global _MODEL

    if _MODEL is None:
        _MODEL, _ = load_model()

    return _MODEL


# ============================================================
# MAIN FEATURE EXTRACTION API
# ============================================================

def extract_features(
    video_path,
    sample_fps=8.0,
    batch_size=16,
):
    """
    Extract temporal visual features from a video.

    Parameters
    ----------
    video_path : str or Path
        Input video file.

    sample_fps : float
        Temporal sampling rate.

    batch_size : int
        GPU inference batch size.

    Returns
    -------
    np.ndarray
        Feature sequence with shape:

            [T, 960]

        T depends on the duration and sampling rate.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    if sample_fps <= 0:
        raise ValueError(
            "sample_fps must be greater than 0."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    model = _get_model()

    features, _, _ = (
        extract_continuous_features(
            model=model,
            video_path=video_path,
            sample_fps=sample_fps,
            batch_size=batch_size,
        )
    )

    if not isinstance(
        features,
        np.ndarray,
    ):
        raise RuntimeError(
            "Feature extractor did not "
            "return a NumPy array."
        )

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D features, "
            f"got shape {features.shape}"
        )

    if features.shape[1] != 960:
        raise RuntimeError(
            f"Expected feature dimension 960, "
            f"got {features.shape[1]}"
        )

    if not np.isfinite(
        features
    ).all():
        raise RuntimeError(
            "Feature sequence contains "
            "NaN or infinite values."
        )

    return features.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# SAVE API
# ============================================================

def extract_and_save(
    video_path,
    output_path,
    sample_fps=8.0,
    batch_size=16,
):
    """
    Extract features and save them as .npy.
    """

    features = extract_features(
        video_path=video_path,
        sample_fps=sample_fps,
        batch_size=batch_size,
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        features,
    )

    return features


__all__ = [
    "extract_features",
    "extract_and_save",
]