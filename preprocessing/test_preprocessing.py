from pathlib import Path

import torch

from video_sampling import load_uniform_frames
from transforms import (
    build_eval_transform,
    build_train_transform,
    preprocess_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
    / "1-wpwyfm1Mg_1.mp4"
)


def main() -> None:
    print(f"Testing: {VIDEO}")

    frames = load_uniform_frames(
        VIDEO,
        num_frames=16,
    )

    print("Raw frames:", frames.shape)
    print("Raw dtype:", frames.dtype)

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_tensors = torch.stack(
        [
            preprocess_frame(frame, train_transform)
            for frame in frames
        ]
    )

    eval_tensors = torch.stack(
        [
            preprocess_frame(frame, eval_transform)
            for frame in frames
        ]
    )

    print("Train tensor:", train_tensors.shape)
    print("Eval tensor:", eval_tensors.shape)

    print("Train dtype:", train_tensors.dtype)
    print("Eval dtype:", eval_tensors.dtype)

    print("Train min:", train_tensors.min().item())
    print("Train max:", train_tensors.max().item())

    print("Preprocessing test successful.")


if __name__ == "__main__":
    main()