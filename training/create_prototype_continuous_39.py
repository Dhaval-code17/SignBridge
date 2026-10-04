from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from inference import extract_features


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLASSES_JSON = (
    ROOT / "configs" / "classes.json"
)

PROTOTYPE_CSV = (
    ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "prototype.csv"
)

MANIFEST_CSV = (
    ROOT
    / "data"
    / "processed"
    / "full_cislr_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "prototype_continuous_39"
)

FEATURE_DIR = (
    OUTPUT_DIR / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

NUM_SEQUENCES = 300

MIN_SIGNS_PER_SEQUENCE = 2
MAX_SIGNS_PER_SEQUENCE = 5

EXPECTED_CLASSES = 39

EXPECTED_FEATURE_DIM = 960

SAMPLE_FPS = 8.0


# ============================================================
# RANDOMNESS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(
    CLASSES_JSON,
    "r",
    encoding="utf-8",
) as f:
    class_data = json.load(f)

class_to_idx = class_data[
    "class_to_idx"
]

classes = list(class_to_idx.keys())

assert len(classes) == EXPECTED_CLASSES, (
    f"Expected {EXPECTED_CLASSES} classes, "
    f"got {len(classes)}"
)

class_set = set(classes)


print("=" * 70)
print("SIGNBRIDGE — 39 CLASS PROTOTYPE SEQUENCE DATASET")
print("=" * 70)

print(
    f"Vocabulary classes: {len(classes)}"
)

print(
    f"Requested sequences: {NUM_SEQUENCES}"
)


# ============================================================
# LOAD PROTOTYPE DATA
# ============================================================

if not PROTOTYPE_CSV.exists():
    raise FileNotFoundError(
        f"Prototype CSV not found:\n{PROTOTYPE_CSV}"
    )

if not MANIFEST_CSV.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_CSV}"
    )


prototype_df = pd.read_csv(
    PROTOTYPE_CSV
)

manifest_df = pd.read_csv(
    MANIFEST_CSV
)


# ============================================================
# NORMALIZE UID
# ============================================================

prototype_df["uid"] = (
    prototype_df["uid"]
    .astype(str)
    .str.strip()
)

manifest_df["uid"] = (
    manifest_df["uid"]
    .astype(str)
    .str.strip()
)

manifest_df["gloss"] = (
    manifest_df["gloss"]
    .astype(str)
    .str.strip()
)


# ============================================================
# BUILD PROTOTYPE UID SET
# ============================================================

prototype_uids = set(
    prototype_df["uid"]
)


# ============================================================
# SELECT ONLY OUR 39 CLASSES
# ============================================================

prototype_manifest = manifest_df[
    manifest_df["uid"].isin(
        prototype_uids
    )
    & manifest_df["gloss"].isin(
        class_set
    )
].copy()


# ============================================================
# VALIDATE ONE VIDEO PER CLASS
# ============================================================

class_counts = (
    prototype_manifest["gloss"]
    .value_counts()
)


print()
print(
    f"Prototype videos found: "
    f"{len(prototype_manifest)}"
)

print(
    f"Prototype classes found: "
    f"{prototype_manifest['gloss'].nunique()}"
)


missing_classes = (
    class_set
    - set(prototype_manifest["gloss"])
)

if missing_classes:
    raise RuntimeError(
        "Missing prototype classes:\n"
        + "\n".join(
            sorted(missing_classes)
        )
    )


duplicate_classes = (
    class_counts[
        class_counts > 1
    ]
)

if len(duplicate_classes) > 0:

    print(
        "Warning: multiple prototype "
        "records found for some classes."
    )

    print(
        duplicate_classes
    )


# ============================================================
# KEEP ONE PROTOTYPE VIDEO PER CLASS
# ============================================================

prototype_manifest = (
    prototype_manifest
    .sort_values(
        ["gloss", "uid"]
    )
    .drop_duplicates(
        subset=["gloss"],
        keep="first",
    )
    .reset_index(drop=True)
)


assert (
    len(prototype_manifest)
    == EXPECTED_CLASSES
), (
    f"Expected {EXPECTED_CLASSES} "
    f"prototype videos, got "
    f"{len(prototype_manifest)}"
)


# ============================================================
# VIDEO PATH RESOLUTION
# ============================================================

def resolve_video_path(
    value
) -> Path:

    path = Path(str(value))

    if path.is_absolute():

        if path.exists():
            return path

    candidates = [
        ROOT / path,

        ROOT
        / "data"
        / "raw"
        / "CISLR"
        / path,

        ROOT
        / "data"
        / "raw"
        / "CISLR"
        / "CISLR_v1.5-a_videos"
        / "CISLR_v1.5-a_videos"
        / path.name,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # Last resort:
    # search by UID/video filename.
    uid = path.stem

    video_root = (
        ROOT
        / "data"
        / "raw"
        / "CISLR"
        / "CISLR_v1.5-a_videos"
        / "CISLR_v1.5-a_videos"
    )

    if video_root.exists():

        matches = list(
            video_root.glob(
                f"{uid}.*"
            )
        )

        if matches:
            return matches[0]

    return path


# ============================================================
# PREPARE PROTOTYPE RECORDS
# ============================================================

records = []

for _, row in prototype_manifest.iterrows():

    video_path = resolve_video_path(
        row["video_path"]
        if "video_path" in row
        else row["uid"]
    )

    if not video_path.exists():

        raise FileNotFoundError(
            "Prototype video not found:\n"
            f"Gloss: {row['gloss']}\n"
            f"UID: {row['uid']}\n"
            f"Path: {video_path}"
        )

    records.append(
        {
            "uid": row["uid"],
            "gloss": row["gloss"],
            "video_path": str(video_path),
        }
    )


# ============================================================
# EXTRACT FEATURES ONCE PER PROTOTYPE
# ============================================================

print()
print("=" * 70)
print("EXTRACTING 39 PROTOTYPE FEATURE SEQUENCES")
print("=" * 70)


feature_cache = {}

for record in records:

    uid = record["uid"]

    gloss = record["gloss"]

    video_path = Path(
        record["video_path"]
    )

    cache_path = (
        FEATURE_DIR
        / f"{uid.replace('/', '_')}.npy"
    )

    print(
        f"\n[{gloss}]"
    )

    print(
        f"Video: {video_path.name}"
    )

    if cache_path.exists():

        features = np.load(
            cache_path
        )

        print(
            f"Loaded cached: "
            f"{features.shape}"
        )

    else:

        features = extract_features(
            str(video_path),
            sample_fps=SAMPLE_FPS,
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        np.save(
            cache_path,
            features,
        )

        print(
            f"Extracted: "
            f"{features.shape}"
        )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2D feature array, "
            f"got {features.shape}"
        )

    if features.shape[1] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"Expected feature dimension "
            f"{EXPECTED_FEATURE_DIM}, "
            f"got {features.shape[1]}"
        )

    if not np.isfinite(
        features
    ).all():

        raise RuntimeError(
            f"Non-finite features for "
            f"{gloss}"
        )

    feature_cache[
        gloss
    ] = {
        "uid": uid,
        "video_path": str(video_path),
        "features": features,
    }


# ============================================================
# CREATE SYNTHETIC CONTINUOUS SEQUENCES
# ============================================================

print()
print("=" * 70)
print("CREATING SYNTHETIC CONTINUOUS SEQUENCES")
print("=" * 70)

sequence_labels = {}

sequence_metadata = []

all_sequence_classes = set()


for sequence_number in range(
    1,
    NUM_SEQUENCES + 1,
):

    sequence_length = random.randint(
        MIN_SIGNS_PER_SEQUENCE,
        MAX_SIGNS_PER_SEQUENCE,
    )

    selected_classes = []

    while len(selected_classes) < sequence_length:

        gloss = random.choice(
            classes
        )

        # Avoid immediate identical signs.
        if (
            selected_classes
            and gloss == selected_classes[-1]
        ):
            continue

        selected_classes.append(
            gloss
        )

    segment_features = []

    segment_lengths = []

    source_uids = []

    source_videos = []

    for gloss in selected_classes:

        item = feature_cache[
            gloss
        ]

        features = item[
            "features"
        ]

        segment_features.append(
            features
        )

        segment_lengths.append(
            int(features.shape[0])
        )

        source_uids.append(
            item["uid"]
        )

        source_videos.append(
            item["video_path"]
        )

        all_sequence_classes.add(
            gloss
        )

    combined = np.concatenate(
        segment_features,
        axis=0,
    ).astype(
        np.float32
    )

    if combined.shape[1] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            "Combined feature dimension "
            "is incorrect."
        )

    sequence_id = (
        f"seq_{sequence_number:06d}"
    )

    feature_path = (
        FEATURE_DIR
        / f"{sequence_id}.npy"
    )

    np.save(
        feature_path,
        combined,
    )

    sequence_labels[
        sequence_id
    ] = selected_classes

    sequence_metadata.append(
        {
            "sequence_id": sequence_id,
            "feature_file": str(
                feature_path.relative_to(ROOT)
            ),
            "labels": selected_classes,
            "source_uids": source_uids,
            "source_videos": source_videos,
            "segment_lengths": segment_lengths,
            "total_frames": int(
                combined.shape[0]
            ),
            "feature_dim": int(
                combined.shape[1]
            ),
            "sample_fps": SAMPLE_FPS,
        }
    )


# ============================================================
# SAVE LABELS
# ============================================================

labels_path = (
    OUTPUT_DIR
    / "labels.json"
)

with open(
    labels_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        sequence_labels,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "dataset_type":
        "synthetic_continuous_prototype",

    "num_sequences":
        NUM_SEQUENCES,

    "num_classes":
        EXPECTED_CLASSES,

    "classes":
        classes,

    "feature_dim":
        EXPECTED_FEATURE_DIM,

    "sample_fps":
        SAMPLE_FPS,

    "min_signs":
        MIN_SIGNS_PER_SEQUENCE,

    "max_signs":
        MAX_SIGNS_PER_SEQUENCE,

    "seed":
        SEED,

    "sequences":
        sequence_metadata,
}


metadata_path = (
    OUTPUT_DIR
    / "metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE CSV INDEX
# ============================================================

sequence_rows = []

for item in sequence_metadata:

    sequence_rows.append(
        {
            "sequence_id":
                item["sequence_id"],

            "feature_file":
                item["feature_file"],

            "labels":
                " | ".join(
                    item["labels"]
                ),

            "num_signs":
                len(item["labels"]),

            "total_frames":
                item["total_frames"],

            "feature_dim":
                item["feature_dim"],

            "sample_fps":
                item["sample_fps"],
        }
    )


sequences_df = pd.DataFrame(
    sequence_rows
)

sequences_csv = (
    OUTPUT_DIR
    / "sequences.csv"
)

sequences_df.to_csv(
    sequences_csv,
    index=False,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

assert len(sequence_labels) == (
    NUM_SEQUENCES
)

assert (
    all_sequence_classes
    == class_set
), (
    "Not every one of the 39 classes "
    "appeared in generated sequences."
)


saved_files = list(
    FEATURE_DIR.glob(
        "seq_*.npy"
    )
)

assert len(saved_files) == (
    NUM_SEQUENCES
)


# Check first few sequences.
for path in saved_files[:10]:

    arr = np.load(path)

    assert arr.ndim == 2
    assert arr.shape[1] == EXPECTED_FEATURE_DIM
    assert np.isfinite(arr).all()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROTOTYPE CONTINUOUS DATASET COMPLETE")
print("=" * 70)

print(
    f"Classes: {EXPECTED_CLASSES}"
)

print(
    f"Sequences: {NUM_SEQUENCES}"
)

print(
    f"Classes covered: "
    f"{len(all_sequence_classes)}/{EXPECTED_CLASSES}"
)

print(
    f"Feature dimension: "
    f"{EXPECTED_FEATURE_DIM}"
)

print(
    f"Sample FPS: "
    f"{SAMPLE_FPS}"
)

print()
print(
    f"Features: {FEATURE_DIR}"
)

print(
    f"Labels:   {labels_path}"
)

print(
    f"Metadata: {metadata_path}"
)

print(
    f"Index:    {sequences_csv}"
)

print()
print(
    "MODEL 2 INPUT CONTRACT:"
)

print(
    "[T, 960] float32 feature sequence"
)

print(
    "LABEL CONTRACT:"
)

print(
    '["gloss_1", "gloss_2", ...]'
)

print("=" * 70)