from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort

from inference import extract_features
from training.extract_continuous_features import (
    TRANSFORM,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
    / "--7-rCNOiK0_1.mp4"
)

ONNX_PATH = (
    ROOT
    / "models"
    / "onnx"
    / "mobilenet_v3_stage2_encoder.onnx"
)

SAMPLE_FPS = 8.0
BATCH_SIZE = 16


# ============================================================
# VIDEO SAMPLING
# ============================================================

def sample_video_frames(
    video_path,
    sample_fps,
):
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:
        raise RuntimeError(
            f"Invalid source FPS: {source_fps}"
        )

    interval = 1.0 / sample_fps
    next_sample_time = 0.0

    frames = []
    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = (
            frame_index / source_fps
        )

        if timestamp + 1e-9 >= next_sample_time:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(frame)

            next_sample_time += interval

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError(
            "No frames sampled."
        )

    return frames


# ============================================================
# ONNX EXTRACTION
# ============================================================

def extract_onnx_features(
    session,
    frames,
):
    batches = []

    for start in range(
        0,
        len(frames),
        BATCH_SIZE,
    ):

        batch_frames = frames[
            start:start + BATCH_SIZE
        ]

        tensors = [
            TRANSFORM(frame)
            for frame in batch_frames
        ]

        batch = torch.stack(
            tensors,
            dim=0,
        )

        batch_np = (
            batch.numpy()
            .astype(np.float32)
        )

        output = session.run(
            ["features"],
            {
                "input": batch_np
            },
        )[0]

        batches.append(
            output
        )

    features = np.concatenate(
        batches,
        axis=0,
    ).astype(np.float32)

    # Match the PyTorch API contract:
    # L2 normalize each temporal vector.
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise RuntimeError(
            "ONNX produced a zero feature vector."
        )

    features = (
        features / norms
    )

    return features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("REAL-VIDEO PYTORCH vs ONNX VERIFICATION")
    print("=" * 65)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found:\n{ONNX_PATH}"
        )

    print(
        f"Video:\n{VIDEO_PATH}"
    )

    print(
        f"Sampling FPS: {SAMPLE_FPS}"
    )

    # --------------------------------------------------------
    # PYTORCH API
    # --------------------------------------------------------

    print(
        "\nRunning PyTorch API..."
    )

    pytorch_features = extract_features(
        VIDEO_PATH,
        sample_fps=SAMPLE_FPS,
        batch_size=BATCH_SIZE,
    )

    print(
        f"PyTorch shape: "
        f"{pytorch_features.shape}"
    )

    # --------------------------------------------------------
    # SAME VIDEO FRAMES
    # --------------------------------------------------------

    frames = sample_video_frames(
        VIDEO_PATH,
        SAMPLE_FPS,
    )

    print(
        f"Sampled frames: {len(frames)}"
    )

    # --------------------------------------------------------
    # ONNX
    # --------------------------------------------------------

    print(
        "\nRunning ONNXRuntime..."
    )

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    onnx_features = extract_onnx_features(
        session,
        frames,
    )

    print(
        f"ONNX shape: "
        f"{onnx_features.shape}"
    )

    # --------------------------------------------------------
    # SHAPE CHECK
    # --------------------------------------------------------

    if pytorch_features.shape != (
        onnx_features.shape
    ):
        raise RuntimeError(
            "PyTorch and ONNX output shapes differ."
        )

    # --------------------------------------------------------
    # NUMERICAL COMPARISON
    # --------------------------------------------------------

    difference = np.abs(
        pytorch_features
        - onnx_features
    )

    max_abs_diff = float(
        difference.max()
    )

    mean_abs_diff = float(
        difference.mean()
    )

    flat_pt = pytorch_features.reshape(-1)
    flat_ox = onnx_features.reshape(-1)

    denominator = (
        np.linalg.norm(flat_pt)
        * np.linalg.norm(flat_ox)
    )

    cosine_similarity = float(
        np.dot(flat_pt, flat_ox)
        / denominator
    )

    print(
        "\nComparison:"
    )

    print(
        f"Max absolute difference : "
        f"{max_abs_diff:.8f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_abs_diff:.8f}"
    )

    print(
        f"Cosine similarity       : "
        f"{cosine_similarity:.8f}"
    )

    # --------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------

    print(
        f"PyTorch finite: "
        f"{np.isfinite(pytorch_features).all()}"
    )

    print(
        f"ONNX finite:    "
        f"{np.isfinite(onnx_features).all()}"
    )

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    if (
        max_abs_diff >= 0.005
        or mean_abs_diff >= 0.0005
        or cosine_similarity <= 0.9999
    ):
        raise RuntimeError(
            "Real-video PyTorch vs ONNX "
            "verification failed."
        )

    print(
        "\n" + "=" * 65
    )

    print(
        "REAL-VIDEO ONNX VERIFICATION PASSED"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":
    main()