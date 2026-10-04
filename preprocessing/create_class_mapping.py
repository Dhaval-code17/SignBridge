import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model1_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "classes.json"
)


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)

    classes = sorted(df["gloss"].unique())

    class_to_idx = {
        label: index
        for index, label in enumerate(classes)
    }

    idx_to_class = {
        str(index): label
        for label, index in class_to_idx.items()
    }

    data = {
        "num_classes": len(classes),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Classes: {len(classes)}")
    print(f"Saved: {OUTPUT_PATH}")

    print("\nMapping:")
    for label, index in class_to_idx.items():
        print(f"{index:02d} -> {label}")


if __name__ == "__main__":
    main()