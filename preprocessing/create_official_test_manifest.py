from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "data" / "raw" / "CISLR" / "test.csv"

VIDEO_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

CLASSES_JSON = ROOT / "configs" / "classes.json"

OUTPUT_CSV = ROOT / "data" / "processed" / "model1_test_manifest.csv"


# ============================================================
# CHECK INPUTS
# ============================================================

for path in [TEST_CSV, VIDEO_ROOT, CLASSES_JSON]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    class_config = json.load(f)

class_to_idx = class_config["class_to_idx"]
allowed_classes = set(class_to_idx.keys())


# ============================================================
# LOAD OFFICIAL TEST CSV
# ============================================================

df = pd.read_csv(TEST_CSV)

required_columns = ["uid", "gloss"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in test.csv: {missing_columns}"
    )


print("=" * 60)
print("CREATING OFFICIAL CISLR TEST MANIFEST")
print("=" * 60)

print(f"Official test rows: {len(df)}")
print(f"Official test columns: {list(df.columns)}")


# ============================================================
# CLEAN VALUES
# ============================================================

df["uid"] = df["uid"].astype(str).str.strip()
df["gloss"] = df["gloss"].astype(str).str.strip()


# ============================================================
# FILTER TO MODEL-1 39 CLASSES
# ============================================================

df = df[df["gloss"].isin(allowed_classes)].copy()

if df.empty:
    raise ValueError(
        "No official test samples matched the 39 Model-1 classes."
    )


# ============================================================
# CHECK DUPLICATE UIDS
# ============================================================

duplicate_uids = df[df["uid"].duplicated(keep=False)]

if not duplicate_uids.empty:
    raise ValueError(
        "Duplicate UIDs found in filtered official test set:\n"
        + duplicate_uids["uid"].to_string(index=False)
    )


# ============================================================
# ADD VIDEO PATH
# ============================================================

df["video_path"] = df["uid"].apply(
    lambda uid: str(VIDEO_ROOT / f"{uid}.mp4")
)


# ============================================================
# VALIDATE VIDEOS
# ============================================================

missing_videos = [
    path
    for path in df["video_path"]
    if not Path(path).exists()
]

if missing_videos:
    print("\nMissing videos:")
    for path in missing_videos[:20]:
        print(path)

    if len(missing_videos) > 20:
        print(
            f"... and {len(missing_videos) - 20} more"
        )

    raise FileNotFoundError(
        f"{len(missing_videos)} video files are missing."
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    ["gloss", "uid"]
).reset_index(drop=True)


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

preferred_columns = [
    "uid",
    "gloss",
    "duration",
    "category",
    "video_path",
]

output_columns = [
    col for col in preferred_columns
    if col in df.columns
]

df = df[output_columns]


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

class_counts = (
    df["gloss"]
    .value_counts()
    .sort_index()
)

missing_classes = sorted(
    allowed_classes - set(df["gloss"])
)

print("\nFiltered official test samples:", len(df))
print("Classes:", df["gloss"].nunique())
print("Videos verified:", len(df) - len(missing_videos))

print("\nClass counts:")
for gloss, count in class_counts.items():
    print(f"{gloss:15s} {count}")

if missing_classes:
    print("\nWARNING - classes missing from official test:")
    for gloss in missing_classes:
        print(f"  {gloss}")
else:
    print("\nAll 39 Model-1 classes are present.")


print("\nSaved manifest:")
print(OUTPUT_CSV)

print("\n" + "=" * 60)
print("OFFICIAL TEST MANIFEST COMPLETE")
print("=" * 60)