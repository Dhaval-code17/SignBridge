from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _uniform_indices(frame_count: int, num_frames: int) -> np.ndarray:
    """Generate ordered frame indices across a decoded video."""
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than 0")

    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0")

    if frame_count == 1:
        return np.zeros(num_frames, dtype=np.int64)

    # Always return exactly num_frames indices.
    indices = np.linspace(
        0,
        frame_count - 1,
        num_frames,
    )

    return np.rint(indices).astype(np.int64)


def load_uniform_frames(
    video_path: str | Path,
    num_frames: int = 16,
) -> np.ndarray:
    """
    Robustly decode a video sequentially and sample frames uniformly.

    This intentionally avoids cv2 random frame seeking because some
    MP4 files can report a frame count correctly but fail on CAP_PROP_POS_FRAMES.

    Returns:
        np.ndarray:
            shape [num_frames, H, W, 3]
            dtype uint8
            RGB
    """
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frames: list[np.ndarray] = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame is None or frame.size == 0:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(frame_rgb)

    finally:
        cap.release()

    actual_frame_count = len(frames)

    if actual_frame_count == 0:
        raise RuntimeError(
            f"No decodable frames found in: {video_path}"
        )

    indices = _uniform_indices(
        actual_frame_count,
        num_frames,
    )

    sampled = [
        frames[int(index)]
        for index in indices
    ]

    return np.stack(
        sampled,
        axis=0,
    )