import os
import json
import shutil


FEATURE_DIR = r"data\continuous\new_train\features"

LABELS_PATH = r"C:\Users\nihav\Downloads\final_model1_features\final_model1_features_train\labels.json"

OUTPUT_DIR = r"data\continuous\new_train"


def main():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    feature_files = sorted(
        f for f in os.listdir(FEATURE_DIR)
        if f.endswith(".npy")
    )

    print("Feature files:", len(feature_files))
    print("Labels:", len(labels))

    if len(feature_files) != len(labels):
        raise ValueError(
            f"Feature/label count mismatch: "
            f"{len(feature_files)} features vs {len(labels)} labels"
        )

    for feature_file in feature_files:
        sequence_id = int(
            os.path.splitext(feature_file)[0].replace("seq_", "")
        )

        if str(sequence_id) not in labels:
            raise ValueError(
                f"Missing label for {feature_file}"
            )

        label = labels[str(sequence_id)]

        label_file = os.path.splitext(feature_file)[0] + ".json"
        label_path = os.path.join(OUTPUT_DIR, label_file)

        with open(label_path, "w", encoding="utf-8") as f:
            json.dump([label], f, ensure_ascii=False)

    print("Created label files:", len(feature_files))
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()