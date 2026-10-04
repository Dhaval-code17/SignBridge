from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.mobilenet_model import MobileNetV3SignRecognizer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_stage2_best.pth"
)

PROTOTYPE_CSV = (
    ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "prototype.csv"
)

VIDEO_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

DEFAULT_OUTPUT = (
    ROOT
    / "models"
    / "features"
    / "feature_sequence.npy"
)

IMAGE_SIZE = 224
DEFAULT_NUM_FRAMES = 16


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
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
# MODEL LOADING
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
        raise ValueError(
            f"Expected feature dimension 960, "
            f"got {feature_dim}"
        )

    return model, checkpoint


# ============================================================
# VIDEO DECODING
# ============================================================

def decode_video(video_path):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded from:\n"
            f"{video_path}"
        )

    return frames


# ============================================================
# UNIFORM SAMPLING
# ============================================================

def sample_frames(
    frames,
    num_frames,
):

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be > 0"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
    ).round().astype(int)

    return [
        frames[int(index)]
        for index in indices
    ]


# ============================================================
# EXTRACT FEATURE SEQUENCE
# ============================================================

@torch.no_grad()
def extract_feature_sequence(
    model,
    video_path,
    num_frames=DEFAULT_NUM_FRAMES,
):

    frames = decode_video(
        video_path
    )

    sampled_frames = sample_frames(
        frames,
        num_frames,
    )

    tensors = [
        TRANSFORM(frame)
        for frame in sampled_frames
    ]

    batch = torch.stack(
        tensors,
        dim=0,
    )

    batch = batch.to(
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

    if features.shape[0] != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} feature vectors, "
            f"got {features.shape[0]}"
        )

    if features.shape[1] != 960:
        raise RuntimeError(
            f"Expected 960 features, "
            f"got {features.shape[1]}"
        )

    # Normalize each frame feature.
    features = F.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu().numpy().astype(
        np.float32
    )


# ============================================================
# AUTOMATIC SMOKE-TEST VIDEO
# ============================================================

def get_default_video():

    if not PROTOTYPE_CSV.exists():
        raise FileNotFoundError(
            f"Prototype CSV not found:\n"
            f"{PROTOTYPE_CSV}"
        )

    import pandas as pd

    df = pd.read_csv(
        PROTOTYPE_CSV
    )

    if df.empty:
        raise ValueError(
            "prototype.csv is empty."
        )

    uid = str(
        df.iloc[0]["uid"]
    ).strip()

    video_path = (
        VIDEO_ROOT
        / f"{uid}.mp4"
    )

    return video_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract MobileNetV3 "
            "[T, 960] feature sequences."
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=(
            "Path to input MP4. "
            "Defaults to first CISLR prototype."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output .npy path.",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of uniformly sampled frames.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("SIGNBRIDGE PERSON 1 FEATURE EXTRACTOR")
    print("=" * 65)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading Stage-2 model..."
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

    print(
        f"Frames requested: "
        f"{args.num_frames}"
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if args.video is None:
        video_path = get_default_video()
    else:
        video_path = Path(
            args.video
        )

    print(
        f"\nInput video:\n"
        f"{video_path}"
    )

    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    print(
        "\nExtracting feature sequence..."
    )

    feature_sequence = extract_feature_sequence(
        model=model,
        video_path=video_path,
        num_frames=args.num_frames,
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
    # SAVE
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        feature_sequence,
    )

    print(
        f"\nSaved feature sequence:\n"
        f"{output_path}"
    )

    # --------------------------------------------------------
    # VERIFY SAVED FILE
    # --------------------------------------------------------

    loaded = np.load(
        output_path
    )

    print(
        f"Reloaded shape: "
        f"{loaded.shape}"
    )

    print(
        f"Reloaded dtype: "
        f"{loaded.dtype}"
    )

    if loaded.shape != (
        args.num_frames,
        960,
    ):
        raise RuntimeError(
            "Saved feature sequence "
            "has an unexpected shape."
        )

    print(
        "\nFeature extraction verified successfully."
    )

    print("=" * 65)


if __name__ == "__main__":
    main()