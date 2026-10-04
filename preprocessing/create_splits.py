from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "model1_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.30


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)

    # Verify that every class has enough examples for a stratified split.
    counts = df["gloss"].value_counts()

    if (counts < 2).any():
        bad_classes = counts[counts < 2]
        raise ValueError(
            "These classes have fewer than 2 samples:\n"
            f"{bad_classes}"
        )

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["gloss"],
        shuffle=True,
    )

    train_df = train_df.sort_values(["gloss", "uid"]).reset_index(drop=True)
    val_df = val_df.sort_values(["gloss", "uid"]).reset_index(drop=True)

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Number of classes: {df['gloss'].nunique()}")

    print("\nTraining class distribution:")
    print(train_df["gloss"].value_counts().sort_index().to_string())

    print("\nValidation class distribution:")
    print(val_df["gloss"].value_counts().sort_index().to_string())

    print(f"\nSaved:")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()