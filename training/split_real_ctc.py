import random
from pathlib import Path


DATA_DIR = Path("data/continuous")

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

VAL_RATIO = 0.2
SEED = 42


def main():
    feature_files = sorted(DATA_DIR.glob("*.npy"))

    if len(feature_files) == 0:
        raise ValueError("No .npy files found in data/continuous")

    random.seed(SEED)
    random.shuffle(feature_files)

    val_size = int(len(feature_files) * VAL_RATIO)

    val_files = feature_files[:val_size]
    train_files = feature_files[val_size:]

    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)

    for feature_file in train_files:
        label_file = feature_file.with_suffix(".json")

        feature_file.rename(
            TRAIN_DIR / feature_file.name
        )

        label_file.rename(
            TRAIN_DIR / label_file.name
        )

    for feature_file in val_files:
        label_file = feature_file.with_suffix(".json")

        feature_file.rename(
            VAL_DIR / feature_file.name
        )

        label_file.rename(
            VAL_DIR / label_file.name
        )

    print("Split complete!")
    print("Training sequences:", len(train_files))
    print("Validation sequences:", len(val_files))


if __name__ == "__main__":
    main()