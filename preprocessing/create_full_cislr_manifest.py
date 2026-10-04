from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CISLR_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "CISLR"
)

DATASET_CSV = (
    CISLR_ROOT
    / "dataset.csv"
)

VIDEO_ROOT = (
    CISLR_ROOT
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "full_cislr_manifest.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

if not DATASET_CSV.exists():
    raise FileNotFoundError(
        f"dataset.csv not found:\n{DATASET_CSV}"
    )

if not VIDEO_ROOT.exists():
    raise FileNotFoundError(
        f"Video directory not found:\n{VIDEO_ROOT}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_CSV
)

required_columns = [
    "uid",
    "gloss",
    "duration",
    "category",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CLEAN VALUES
# ============================================================

df["uid"] = (
    df["uid"]
    .astype(str)
    .str.strip()
)

df["gloss"] = (
    df["gloss"]
    .astype(str)
    .str.strip()
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

if df["uid"].duplicated().any():

    duplicates = df[
        df["uid"].duplicated(
            keep=False
        )
    ]

    raise ValueError(
        "Duplicate UIDs found."
        + "\n"
        + duplicates[
            "uid"
        ].to_string(
            index=False
        )
    )


# ============================================================
# VIDEO PATH
# ============================================================

df["video_path"] = df[
    "uid"
].apply(
    lambda uid:
    str(
        VIDEO_ROOT
        / f"{uid}.mp4"
    )
)


# ============================================================
# VERIFY ALL VIDEOS
# ============================================================

print(
    "Checking all video files..."
)

missing_videos = [
    path
    for path in df[
        "video_path"
    ]
    if not Path(path).exists()
]

if missing_videos:

    print(
        "\nFirst missing files:"
    )

    for path in missing_videos[:20]:
        print(path)

    raise FileNotFoundError(
        f"{len(missing_videos)} "
        "video files are missing."
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "gloss",
        "uid",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

class_counts = (
    df["gloss"]
    .value_counts()
)

print(
    "\n" + "=" * 65
)

print(
    "FULL CISLR MANIFEST"
)

print(
    "=" * 65
)

print(
    f"Videos: "
    f"{len(df)}"
)

print(
    f"Unique glosses: "
    f"{df['gloss'].nunique()}"
)

print(
    f"Videos verified: "
    f"{len(df) - len(missing_videos)}"
)

print(
    f"Minimum videos/class: "
    f"{class_counts.min()}"
)

print(
    f"Maximum videos/class: "
    f"{class_counts.max()}"
)

print(
    f"Median videos/class: "
    f"{class_counts.median()}"
)

print(
    "\nClass frequency distribution:"
)

print(
    class_counts.describe()
)

print(
    "\nSaved:"
)

print(
    OUTPUT_CSV
)

print(
    "\n" + "=" * 65
)

print(
    "FULL CISLR MANIFEST COMPLETE"
)

print(
    "=" * 65
)