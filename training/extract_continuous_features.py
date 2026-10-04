from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.mobilenet_model import MobileNetV3SignRecognizer


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_full_vocab_contrastive_best.pth"
)

DEFAULT_OUTPUT = (
    ROOT
    / "models"
    / "features"
    / "continuous_feature_sequence.npy"
)

DEFAULT_METADATA = (
    ROOT
    / "models"
    / "features"
    / "continuous_feature_metadata.json"
)

IMAGE_SIZE = 224
DEFAULT_SAMPLE_FPS = 8.0
DEFAULT_BATCH_SIZE = 16


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    class_to_idx = checkpoint[
        "class_to_idx"
    ]

    num_classes = len(class_to_idx)

    model = MobileNetV3SignRecognizer(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    feature_dim = checkpoint[
        "feature_dim"
    ]

    if feature_dim != 960:
        raise RuntimeError(
            f"Expected feature dimension 960, "
            f"got {feature_dim}"
        )

    return model, checkpoint


# ============================================================
# READ VIDEO METADATA
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if source_fps <= 0:
        raise RuntimeError(
            f"Invalid source FPS: {source_fps}"
        )

    duration = (
        frame_count / source_fps
    )

    return {
        "source_fps": float(source_fps),
        "source_frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": float(duration),
    }


# ============================================================
# DECODE AT TARGET SAMPLING RATE
# ============================================================

def decode_sampled_frames(
    video_path,
    sample_fps,
):
    """
    Sequentially decode the video and select frames
    at approximately sample_fps.

    Returns:
        frames: list[np.ndarray]
        timestamps: list[float]
        source_fps: float
    """

    if sample_fps <= 0:
        raise ValueError(
            "sample_fps must be greater than 0."
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:
        cap.release()

        raise RuntimeError(
            f"Invalid source FPS: "
            f"{source_fps}"
        )

    interval = 1.0 / sample_fps

    next_sample_time = 0.0

    frames = []
    timestamps = []

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
            timestamps.append(
                float(timestamp)
            )

            next_sample_time += interval

        frame_index += 1

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No sampled frames obtained from:\n"
            f"{video_path}"
        )

    return (
        frames,
        timestamps,
        float(source_fps),
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_continuous_features(
    model,
    video_path,
    sample_fps,
    batch_size,
):
    """
    Returns a temporal feature sequence:

        [T, 960]

    where T depends on video duration and sampling FPS.
    """

    frames, timestamps, source_fps = (
        decode_sampled_frames(
            video_path,
            sample_fps,
        )
    )

    all_features = []

    for start in range(
        0,
        len(frames),
        batch_size,
    ):

        batch_frames = frames[
            start:start + batch_size
        ]

        tensors = [
            TRANSFORM(frame)
            for frame in batch_frames
        ]

        batch = torch.stack(
            tensors,
            dim=0,
        ).to(
            DEVICE,
            non_blocking=True,
        )

        features = model.extract_features(
            batch
        )

        if features.ndim != 2:
            raise RuntimeError(
                f"Unexpected feature shape: "
                f"{tuple(features.shape)}"
            )

        if features.shape[1] != 960:
            raise RuntimeError(
                f"Expected 960 features, "
                f"got {features.shape[1]}"
            )

        # Normalize every temporal feature vector.
        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        all_features.append(
            features.cpu()
        )

    feature_sequence = torch.cat(
        all_features,
        dim=0,
    ).numpy().astype(
        np.float32
    )

    if feature_sequence.shape[0] != len(
        timestamps
    ):
        raise RuntimeError(
            "Number of feature vectors does not "
            "match number of sampled timestamps."
        )

    if feature_sequence.shape[1] != 960:
        raise RuntimeError(
            "Final feature dimension is not 960."
        )

    return (
        feature_sequence,
        timestamps,
        source_fps,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract continuous SignBridge "
            "[T,960] MobileNetV3 features."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Input video path.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        type=str,
        help="Output .npy feature path.",
    )

    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        type=str,
        help="Output metadata JSON path.",
    )

    parser.add_argument(
        "--sample-fps",
        default=DEFAULT_SAMPLE_FPS,
        type=float,
        help="Temporal sampling rate.",
    )

    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="GPU inference batch size.",
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Input video not found:\n"
            f"{video_path}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    if args.sample_fps <= 0:
        raise ValueError(
            "sample_fps must be greater than 0."
        )

    print("=" * 65)
    print("SIGNBRIDGE CONTINUOUS FEATURE EXTRACTOR")
    print("=" * 65)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Input video:\n{video_path}"
    )

    print(
        f"Sampling FPS: {args.sample_fps}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading final 4764-class contrastive encoder..."
    )

    model, checkpoint = load_model()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Feature dimension: "
        f"{checkpoint['feature_dim']}"
    )

    # --------------------------------------------------------
    # SOURCE VIDEO INFO
    # --------------------------------------------------------

    info = get_video_info(
        video_path
    )

    print(
        "\nSource video:"
    )

    print(
        f"  FPS: "
        f"{info['source_fps']:.3f}"
    )

    print(
        f"  Frames: "
        f"{info['source_frame_count']}"
    )

    print(
        f"  Resolution: "
        f"{info['width']}x{info['height']}"
    )

    print(
        f"  Duration: "
        f"{info['duration_seconds']:.3f}s"
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    print(
        "\nExtracting continuous features..."
    )

    feature_sequence, timestamps, source_fps = (
        extract_continuous_features(
            model=model,
            video_path=video_path,
            sample_fps=args.sample_fps,
            batch_size=args.batch_size,
        )
    )

    print(
        f"Sampled frames: "
        f"{len(timestamps)}"
    )

    print(
        f"Feature shape: "
        f"{feature_sequence.shape}"
    )

    print(
        f"Feature dtype: "
        f"{feature_sequence.dtype}"
    )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    metadata_path = Path(
        args.metadata
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------------

    np.save(
        output_path,
        feature_sequence,
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "video_path": str(video_path),
        "feature_shape": list(
            feature_sequence.shape
        ),
        "feature_dim": int(
            feature_sequence.shape[1]
        ),
        "num_feature_frames": int(
            feature_sequence.shape[0]
        ),
        "sample_fps": float(
            args.sample_fps
        ),
        "source_fps": float(
            source_fps
        ),
        "source_frame_count": int(
            info["source_frame_count"]
        ),
        "duration_seconds": float(
            info["duration_seconds"]
        ),
        "image_size": IMAGE_SIZE,
        "normalization": {
            "mean": [
                0.485,
                0.456,
                0.406,
            ],
            "std": [
                0.229,
                0.224,
                0.225,
            ],
        },
        "model": "MobileNetV3-Large",
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # RELOAD VERIFICATION
    # --------------------------------------------------------

    loaded_features = np.load(
        output_path
    )

    if loaded_features.shape != (
        feature_sequence.shape
    ):

        raise RuntimeError(
            "Reloaded feature shape mismatch."
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        loaded_metadata = json.load(f)

    if loaded_metadata[
        "feature_dim"
    ] != 960:

        raise RuntimeError(
            "Metadata feature dimension is not 960."
        )

    print(
        "\nSaved:"
    )

    print(
        f"  Features: {output_path}"
    )

    print(
        f"  Metadata: {metadata_path}"
    )

    print(
        "\nVerification:"
    )

    print(
        f"  Reloaded feature shape: "
        f"{loaded_features.shape}"
    )

    print(
        f"  Metadata feature dimension: "
        f"{loaded_metadata['feature_dim']}"
    )

    print(
        f"  Metadata sampling FPS: "
        f"{loaded_metadata['sample_fps']}"
    )

    print(
        "\nContinuous feature extraction "
        "verified successfully."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()

