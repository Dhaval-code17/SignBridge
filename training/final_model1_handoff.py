from __future__ import annotations

"""
SignBridge - Final Model 1 Continuous Feature Handoff

Purpose
-------
1. Load the final 4,764-class MobileNetV3-Large contrastive checkpoint.
2. Extract ordered continuous visual features from isolated CISLR videos.
3. Save each video as a [T, 960] float32 .npy sequence for Model 2.
4. Save labels + metadata and run strict interface validation.

This script does NOT train the model and does NOT run the Transformer/CTC model.
It validates the Model-1 -> Model-2 contract so Person 2 can consume the output.

Expected Model 1 contract
-------------------------
Input : video frames
Output: [T, 960] float32, finite, ordered by time
Sampling: 8 FPS by default

Example from project root (PowerShell)
--------------------------------------
python -m training.final_model1_handoff extract `
  --csv data/processed/final_val.csv `
  --video-dir data/raw/CISLR/CISLR_v1.5-a_videos/CISLR_v1.5-a_videos `
  --checkpoint models/checkpoints/mobilenet_v3_full_vocab_contrastive_best.pth `
  --output data/processed/final_model1_features_val `
  --limit 50

Then validate:
python -m training.final_model1_handoff verify `
  --features data/processed/final_model1_features_val `
  --classes configs/classes_final.json

Notes
-----
- Pass the actual final split CSV path if its filename differs.
- Use --limit for a fast integration smoke test. Remove it for a full split.
- The checkpoint's 960-D backbone features are used; the 128-D contrastive
  projection head is intentionally NOT part of the Model-2 interface.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm


DEFAULT_SAMPLE_FPS = 8.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
EXPECTED_FEATURE_DIM = 960
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ContrastiveMobileNetBackbone(nn.Module):
    """MobileNetV3-Large backbone returning the 960-D pre-projection vector."""

    def __init__(self) -> None:
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_classes(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports either {"classes": [...]}, {"mapping": {...}}, or a direct mapping.
    if isinstance(data, dict) and isinstance(data.get("class_to_idx"), dict):
        mapping = data["class_to_idx"]
    elif isinstance(data, dict) and isinstance(data.get("mapping"), dict):
        mapping = data["mapping"]
    elif isinstance(data, dict) and isinstance(data.get("classes"), list):
        mapping = {str(name): i for i, name in enumerate(data["classes"])}
    elif isinstance(data, dict):
        mapping = data
    else:
        raise ValueError(f"Unsupported classes JSON format: {path}")

    normalized: dict[str, int] = {}
    for k, v in mapping.items():
        try:
            normalized[str(k).strip()] = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class mapping entry: {k!r}: {v!r}") from exc

    return normalized


def clean_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null"}:
        return None
    return text


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lower_to_real = {str(c).strip().lower(): str(c) for c in df.columns}
    for candidate in candidates:
        hit = lower_to_real.get(candidate.lower())
        if hit is not None:
            return hit
    if required:
        raise ValueError(
            "Could not find a required column. Tried: "
            + ", ".join(candidates)
            + f". Available columns: {list(df.columns)}"
        )
    return None


def resolve_video_path(row: pd.Series, video_dir: Path) -> Path:
    # Prefer an explicit path/filename field when present.
    path_candidates = [
        "video_path",
        "filepath",
        "file_path",
        "video_file",
        "filename",
        "file",
        "path",
    ]
    for col in path_candidates:
        if col in row.index:
            raw = row[col]
            if isinstance(raw, str) and raw.strip():
                candidate = Path(raw.strip())
                if candidate.is_absolute() and candidate.exists():
                    return candidate
                candidate2 = video_dir / raw.strip()
                if candidate2.exists():
                    return candidate2

    uid_col = None
    for c in ("uid", "video_uid", "video_id", "id", "youtube_id"):
        if c in row.index:
            uid_col = c
            break
    if uid_col is None:
        # Case-insensitive fallback.
        lower = {str(c).lower(): c for c in row.index}
        for c in ("uid", "video_uid", "video_id", "id", "youtube_id"):
            if c in lower:
                uid_col = lower[c]
                break

    if uid_col is None:
        raise ValueError(
            "CSV needs one of: video_path/filepath/filename OR uid/video_uid/video_id/id."
        )

    uid = clean_label(row[uid_col])
    if uid is None:
        raise ValueError("Encountered a row without a usable video UID.")

    # Most CISLR files use <uid>.mp4. A recursive fallback handles files stored
    # one or more folders below the requested video directory.
    direct = video_dir / f"{uid}.mp4"
    if direct.exists():
        return direct

    direct_any = video_dir / uid
    if direct_any.exists():
        return direct_any

    matches = list(video_dir.rglob(f"{uid}.mp4"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Video not found for UID {uid!r} under {video_dir}")


def sample_video_frames(video_path: Path, sample_fps: float) -> list[np.ndarray]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: list[np.ndarray] = []
    decoded: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(frame)
    finally:
        cap.release()

    if not decoded:
        raise RuntimeError(f"No frames decoded from {video_path}")

    if native_fps <= 0 or not math.isfinite(native_fps):
        native_fps = sample_fps

    if sample_fps >= native_fps:
        indices = np.arange(len(decoded), dtype=np.int64)
    else:
        duration = len(decoded) / native_fps
        target_count = max(1, int(round(duration * sample_fps)))
        positions = np.linspace(0, len(decoded) - 1, target_count, dtype=np.float64)
        indices = np.unique(np.rint(positions).astype(np.int64))

    for idx in indices:
        frame = decoded[int(idx)]
        # OpenCV BGR -> RGB.
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return frames


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load only the trained 960-D MobileNetV3 backbone weights.

    The final contrastive checkpoint contains separate backbone and projection
    weights. Model 2 must receive the 960-D pre-projection encoder output, so
    classifier/projection weights are intentionally ignored.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary saved with torch.save().")

    state: dict[str, torch.Tensor] | None = None

    # Prefer the explicitly saved backbone state from the contrastive checkpoint.
    for key in ("backbone_state_dict", "encoder_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            state = value
            break

    # Fallback: recover backbone weights from the complete model state.
    if state is None:
        value = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if isinstance(value, dict) and value:
            state = value

    if state is None:
        raise KeyError(
            "Checkpoint does not contain backbone_state_dict, "
            "encoder_state_dict, model_state_dict, or state_dict."
        )

    target_keys = set(model.state_dict().keys())
    mapped: dict[str, torch.Tensor] = {}

    for raw_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue

        key = raw_key

        # Remove common wrappers added by DataParallel or model containers.
        for prefix in ("module.", "model.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]

        # Convert the training model's `backbone.<sequential-index>...`
        # convention to this handoff model's `features.<sequential-index>...`.
        if key.startswith("backbone.features."):
            key = "features." + key[len("backbone.features."):]
        elif key.startswith("backbone."):
            key = "features." + key[len("backbone."):]
        elif key.startswith("features."):
            pass
        elif key.startswith("avgpool."):
            pass
        elif key and key[0].isdigit():
            key = "features." + key
        else:
            # Ignore classifier/projection/optimizer-style entries.
            continue

        if key in target_keys:
            mapped[key] = value

    missing = sorted(target_keys - set(mapped))
    unexpected = sorted(set(mapped) - target_keys)

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            "Could not load the complete 960-D MobileNetV3 backbone. "
            f"Missing {len(missing)} model keys; first keys: {preview}"
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected mapped backbone keys: {unexpected[:10]}"
        )

    # Strict loading prevents a partially loaded backbone from silently passing.
    incompatible = model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Backbone state loading was not exact: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    return {
        "checkpoint_keys": len(state),
        "loaded_parameters": len(mapped),
        "missing_keys": [],
        "unexpected_keys": [],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "feature_dim": checkpoint.get("feature_dim"),
        "projection_dim": checkpoint.get("projection_dim"),
        "num_classes": checkpoint.get("num_classes"),
    }


def embed_frames(
    model: nn.Module,
    frames: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            tensors = []
            for frame in batch_frames:
                pil = Image.fromarray(frame)
                tensors.append(EVAL_TRANSFORM(pil))
            batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            features = model(batch)
            if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_DIM:
                raise RuntimeError(
                    f"Unexpected encoder output shape {tuple(features.shape)}; expected [B, {EXPECTED_FEATURE_DIM}]."
                )
            features = features.float().cpu().numpy()
            outputs.append(features)
    result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("Encoder produced NaN/Inf values.")
    return result


def build_records(csv_path: Path, video_dir: Path, classes: dict[str, int]) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    label_col = find_column(
        df,
        ("gloss", "label", "class", "sign", "target", "word"),
        required=True,
    )

    records: list[dict[str, Any]] = []
    missing_labels = 0
    unknown_labels: set[str] = set()
    missing_videos: list[str] = []

    for row_idx, row in df.iterrows():
        label = clean_label(row[label_col])
        if label is None:
            missing_labels += 1
            continue
        if label not in classes:
            unknown_labels.add(label)
            continue

        try:
            video_path = resolve_video_path(row, video_dir)
        except Exception as exc:
            missing_videos.append(f"row={row_idx}: {exc}")
            continue

        uid = None
        for c in ("uid", "video_uid", "video_id", "id", "youtube_id"):
            if c in row.index:
                uid = clean_label(row[c])
                if uid:
                    break
        if uid is None:
            uid = video_path.stem

        records.append(
            {
                "row_index": int(row_idx),
                "uid": uid,
                "gloss": label,
                "video_path": str(video_path),
            }
        )

    if unknown_labels:
        sample = sorted(unknown_labels)[:20]
        raise ValueError(f"Found {len(unknown_labels)} labels not present in classes JSON. Sample: {sample}")
    if missing_videos:
        preview = "\n".join(missing_videos[:10])
        raise FileNotFoundError(
            f"Could not resolve {len(missing_videos)} video files. First entries:\n{preview}"
        )
    if missing_labels:
        print(f"Warning: skipped {missing_labels} unlabeled rows.")
    return records


def extract(args: argparse.Namespace) -> None:
    project_root = Path.cwd()
    csv_path = Path(args.csv)
    video_dir = Path(args.video_dir)
    checkpoint_path = Path(args.checkpoint)
    classes_path = Path(args.classes)
    output_dir = Path(args.output)

    for p, name in (
        (csv_path, "CSV"),
        (video_dir, "video directory"),
        (checkpoint_path, "checkpoint"),
        (classes_path, "classes JSON"),
    ):
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    classes = load_classes(classes_path)
    if len(classes) != 4764:
        raise ValueError(f"Expected 4,764 classes, found {len(classes)} in {classes_path}")

    records = build_records(csv_path, video_dir, classes)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        records = records[: args.limit]

    output_features = output_dir / "features"
    output_features.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 70)
    print("SIGNBRIDGE FINAL MODEL 1 CONTINUOUS FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Device: {device}")
    print(f"CSV: {csv_path}")
    print(f"Video dir: {video_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Vocabulary: {len(classes)} classes")
    print(f"Videos to process: {len(records)}")
    print(f"Sample FPS: {args.sample_fps}")
    print(f"Feature contract: [T, {EXPECTED_FEATURE_DIM}] float32")

    model = ContrastiveMobileNetBackbone().to(device).eval()
    load_info = load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded backbone parameters: {load_info['loaded_parameters']}")
    if load_info["missing_keys"]:
        print(f"Warning: missing backbone keys: {len(load_info['missing_keys'])}")

    sequence_manifest: list[dict[str, Any]] = []
    label_manifest: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []

    for seq_idx, record in enumerate(tqdm(records, desc="Extracting features")):
        try:
            frames = sample_video_frames(Path(record["video_path"]), args.sample_fps)
            features = embed_frames(model, frames, device, args.batch_size)

            if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_DIM:
                raise RuntimeError(f"Bad saved shape: {features.shape}")
            if features.shape[0] <= 0:
                raise RuntimeError("Zero-frame sequence")

            filename = f"seq_{seq_idx:05d}.npy"
            np.save(output_features / filename, features)

            item = {
                "sequence_id": seq_idx,
                "feature_file": f"features/{filename}",
                "uid": record["uid"],
                "gloss": record["gloss"],
                "num_frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "sample_fps": float(args.sample_fps),
                "source_video": record["video_path"],
            }
            sequence_manifest.append(item)
            label_manifest[str(seq_idx)] = record["gloss"]
        except Exception as exc:
            skipped.append({"uid": str(record["uid"]), "gloss": str(record["gloss"]), "error": str(exc)})
            print(f"\n[SKIP] {record['uid']} / {record['gloss']}: {exc}")

    metadata = {
        "project": "SignBridge",
        "model": "MobileNetV3-Large contrastive encoder",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 3,
        "vocabulary_size": len(classes),
        "feature_dim": EXPECTED_FEATURE_DIM,
        "feature_dtype": "float32",
        "feature_shape": "[T,960]",
        "sample_fps": float(args.sample_fps),
        "num_sequences_requested": len(records),
        "num_sequences_written": len(sequence_manifest),
        "num_skipped": len(skipped),
        "label_format": "one gloss per source video",
        "model2_contract": {
            "input": "ordered [T,960] float32 feature sequence",
            "labels": "gloss strings from the 4,764-class vocabulary",
            "classifier_logits": False,
            "projection_head_128d": False,
        },
        "loader": load_info,
        "sequences": sequence_manifest,
        "skipped": skipped,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(label_manifest, f, indent=2, ensure_ascii=False)
    pd.DataFrame(sequence_manifest).to_csv(output_dir / "sequences.csv", index=False)

    print("\n" + "=" * 70)
    print("EXTRACTION RESULT")
    print("=" * 70)
    print(f"Written sequences: {len(sequence_manifest)}")
    print(f"Skipped sequences: {len(skipped)}")
    print(f"Feature directory: {output_features}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    print(f"Labels:   {output_dir / 'labels.json'}")
    print(f"Manifest: {output_dir / 'sequences.csv'}")
    if skipped:
        print("WARNING: Some videos were skipped; do not call this split complete until they are fixed.")


def verify(args: argparse.Namespace) -> None:
    feature_dir = Path(args.features)
    classes_path = Path(args.classes)
    metadata_path = feature_dir / "metadata.json"
    labels_path = feature_dir / "labels.json"
    sequences_csv = feature_dir / "sequences.csv"

    for p in (feature_dir, metadata_path, labels_path, sequences_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required handoff file not found: {p}")

    classes = load_classes(classes_path)
    if len(classes) != 4764:
        raise ValueError(f"Expected 4,764 classes, found {len(classes)}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    manifest = pd.read_csv(sequences_csv)

    errors: list[str] = []

    if metadata.get("feature_dim") != EXPECTED_FEATURE_DIM:
        errors.append(f"metadata feature_dim={metadata.get('feature_dim')}, expected {EXPECTED_FEATURE_DIM}")
    if metadata.get("feature_shape") != "[T,960]":
        errors.append(f"metadata feature_shape={metadata.get('feature_shape')}, expected [T,960]")
    if int(metadata.get("vocabulary_size", -1)) != 4764:
        errors.append(f"metadata vocabulary_size={metadata.get('vocabulary_size')}, expected 4764")

    if len(manifest) != len(labels):
        errors.append(f"manifest rows={len(manifest)} != labels entries={len(labels)}")

    covered: set[str] = set()
    total_frames = 0
    for _, row in manifest.iterrows():
        rel = str(row["feature_file"])
        path = feature_dir / Path(rel)
        if not path.exists():
            errors.append(f"missing feature file: {path}")
            continue

        arr = np.load(path)
        if arr.ndim != 2:
            errors.append(f"{rel}: ndim={arr.ndim}, expected 2")
        elif arr.shape[1] != EXPECTED_FEATURE_DIM:
            errors.append(f"{rel}: shape={arr.shape}, expected [T,960]")
        if arr.dtype != np.float32:
            errors.append(f"{rel}: dtype={arr.dtype}, expected float32")
        if not np.isfinite(arr).all():
            errors.append(f"{rel}: contains NaN/Inf")
        if arr.shape[0] <= 0:
            errors.append(f"{rel}: zero time dimension")

        gloss = clean_label(row.get("gloss"))
        if gloss is None:
            errors.append(f"{rel}: missing gloss")
        elif gloss not in classes:
            errors.append(f"{rel}: unknown gloss {gloss!r}")
        else:
            covered.add(gloss)

        total_frames += int(arr.shape[0])

    skipped = metadata.get("skipped", [])
    if skipped:
        errors.append(f"metadata reports {len(skipped)} skipped videos")

    print("=" * 70)
    print("SIGNBRIDGE MODEL 1 -> MODEL 2 HANDOFF VERIFICATION")
    print("=" * 70)
    print(f"Sequences: {len(manifest)}")
    print(f"Classes in vocabulary: {len(classes)}")
    print(f"Classes represented: {len(covered)}")
    print(f"Total extracted frames: {total_frames}")
    print(f"Feature contract: [T,960] float32")
    print(f"NaN/Inf check: {'PASS' if not errors else 'SEE ERRORS'}")

    if errors:
        print("\nFAIL")
        for e in errors[:50]:
            print(f"- {e}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more")
        raise SystemExit(1)

    print("\nPASS")
    print("All feature files satisfy the Model 2 input contract.")
    print("Model 2 should consume each .npy as an ordered [T,960] float32 sequence.")
    print("No 128-D projection vector or classifier logits should be passed to Model 2.")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="Extract [T,960] features from a CSV split.")
    p_ext.add_argument("--csv", required=True, help="Final split CSV containing gloss + UID/path information.")
    p_ext.add_argument("--video-dir", required=True, help="Directory containing CISLR .mp4 videos.")
    p_ext.add_argument(
        "--checkpoint",
        default="models/checkpoints/mobilenet_v3_full_vocab_contrastive_best.pth",
        help="Final contrastive checkpoint.",
    )
    p_ext.add_argument(
        "--classes",
        default="configs/classes_final.json",
        help="4,764-class mapping JSON.",
    )
    p_ext.add_argument(
        "--output",
        default="data/processed/final_model1_features",
        help="Output directory.",
    )
    p_ext.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    p_ext.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_ext.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a smoke test.")
    p_ext.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    p_ext.set_defaults(func=extract)

    p_ver = sub.add_parser("verify", help="Validate a previously generated Model-1 handoff.")
    p_ver.add_argument("--features", required=True, help="Output directory created by extract.")
    p_ver.add_argument(
        "--classes",
        default="configs/classes_final.json",
        help="4,764-class mapping JSON.",
    )
    p_ver.set_defaults(func=verify)

    return parser


# Used only to keep a clear constant in type-checking/doc tooling.
EXPECTED_CLASS_COUNT = 4764


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)

