from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "full_cislr_manifest.csv"
)

TRAIN_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "final_dev_train.csv"
)

VAL_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "final_dev_val.csv"
)

RANDOM_SEED = 42


# ============================================================
# CHECK INPUT
# ============================================================

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Full manifest not found:\n{MANIFEST}"
    )


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(MANIFEST)

required_columns = [
    "uid",
    "gloss",
    "video_path",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN
# ============================================================

# Keep the original rows so unlabeled videos are not lost.
df["uid"] = df["uid"].astype(str).str.strip()

# Identify rows without a real gloss BEFORE converting to string.
unlabeled_mask = (
    df["gloss"].isna()
    | df["gloss"].astype(str).str.strip().eq("")
    | df["gloss"].astype(str).str.strip().str.lower().eq("nan")
)

unlabeled_df = df.loc[unlabeled_mask].copy()

# Supervised dataset = only genuinely labeled videos.
df = df.loc[~unlabeled_mask].copy()

df["gloss"] = (
    df["gloss"]
    .astype(str)
    .str.strip()
)

# Final vocabulary must be exactly 4,764 glosses.
assert (
    df["gloss"].nunique() == 4764
), (
    f"Expected 4764 supervised glosses, "
    f"got {df['gloss'].nunique()}"
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

if df["uid"].duplicated().any():
    raise ValueError(
        "Duplicate UID detected."
    )


# ============================================================
# DETERMINISTIC SPLIT
# ============================================================

train_parts = []
val_parts = []

for gloss, group in df.groupby(
    "gloss",
    sort=True,
):

    group = group.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    count = len(group)

    if count == 1:

        # Singleton class:
        # keep the only example in training.
        train_parts.append(group)

    else:

        # Repeated class:
        # hold out exactly one example.
        val_parts.append(
            group.iloc[[0]]
        )

        train_parts.append(
            group.iloc[1:]
        )


# ============================================================
# COMBINE
# ============================================================

train_df = pd.concat(
    train_parts,
    ignore_index=True,
)

if val_parts:
    val_df = pd.concat(
        val_parts,
        ignore_index=True,
    )
else:
    val_df = pd.DataFrame(
        columns=df.columns
    )


# ============================================================
# SORT
# ============================================================

train_df = train_df.sort_values(
    ["gloss", "uid"]
).reset_index(drop=True)

val_df = val_df.sort_values(
    ["gloss", "uid"]
).reset_index(drop=True)


# ============================================================
# SAFETY CHECKS
# ============================================================

all_ids = set(
    df["uid"]
)

train_ids = set(
    train_df["uid"]
)

val_ids = set(
    val_df["uid"]
)

if train_ids & val_ids:
    raise RuntimeError(
        "Train/validation UID overlap detected."
    )

if train_ids | val_ids != all_ids:
    raise RuntimeError(
        "Train + validation does not cover "
        "the full dataset."
    )

if len(train_df) + len(val_df) != len(df):
    raise RuntimeError(
        "Split size mismatch."
    )


# Every validation class must appear in train.
train_classes = set(
    train_df["gloss"]
)

val_classes = set(
    val_df["gloss"]
)

missing_train_classes = (
    val_classes - train_classes
)

if missing_train_classes:
    raise RuntimeError(
        "Validation contains classes absent "
        "from training:\n"
        + "\n".join(
            sorted(missing_train_classes)
        )
    )


# ============================================================
# SAVE
# ============================================================

TRAIN_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
)

val_df.to_csv(
    VAL_OUTPUT,
    index=False,
)

UNLABELED_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "final_unlabeled.csv"
)

unlabeled_df.to_csv(
    UNLABELED_OUTPUT,
    index=False,
)

print(
    f"\nUnlabeled videos retained separately: "
    f"{len(unlabeled_df)}"
)

print(
    f"  UNLABELED: {UNLABELED_OUTPUT}"
)


# ============================================================
# SUMMARY
# ============================================================

train_counts = (
    train_df["gloss"]
    .value_counts()
)

val_counts = (
    val_df["gloss"]
    .value_counts()
)

print("=" * 65)
print("FINAL FULL-VOCABULARY DEVELOPMENT SPLIT")
print("=" * 65)

print(
    f"Supervised videos: "
    f"{len(df)}"
)

print(
    f"Original manifest videos: "
    f"{len(df) + len(unlabeled_df)}"
)

print(
    f"Total glosses: "
    f"{df['gloss'].nunique()}"
)

print(
    f"\nTraining videos: "
    f"{len(train_df)}"
)

print(
    f"Training glosses: "
    f"{train_df['gloss'].nunique()}"
)

print(
    f"\nValidation videos: "
    f"{len(val_df)}"
)

print(
    f"Validation glosses: "
    f"{val_df['gloss'].nunique()}"
)

print(
    "\nSingleton glosses kept in training:"
)

singleton_count = (
    (df["gloss"].value_counts() == 1)
    .sum()
)

print(
    singleton_count
)

print(
    "\nRepeated glosses with validation:"
)

print(
    (df["gloss"].value_counts() >= 2)
    .sum()
)

print(
    "\nTrain/validation UID overlap:"
)

print(
    len(train_ids & val_ids)
)

print(
    "\nCoverage:"
)

print(
    len(train_ids | val_ids),
    "/",
    len(all_ids)
)

print(
    "\nSaved:"
)

print(
    f"  TRAIN: {TRAIN_OUTPUT}"
)

print(
    f"  VAL:   {VAL_OUTPUT}"
)

print(
    "\n" + "=" * 65
)

print(
    "DEVELOPMENT SPLIT COMPLETE"
)

print(
    "=" * 65
)