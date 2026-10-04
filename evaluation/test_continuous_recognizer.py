import os
import json
import numpy as np

from inference.continuous_recognizer import ContinuousRecognizer


VAL_DIR = "data/continuous/val"


def main():
    recognizer = ContinuousRecognizer()

    files = sorted(
        f for f in os.listdir(VAL_DIR)
        if f.endswith(".npy")
    )

    for feature_file in files[:10]:
        feature_path = os.path.join(
            VAL_DIR,
            feature_file
        )

        label_file = feature_file.replace(
            ".npy",
            ".json"
        )

        with open(
            os.path.join(VAL_DIR, label_file),
            "r",
            encoding="utf-8",
        ) as f:
            labels = json.load(f)

        features = np.load(feature_path)

        prediction = recognizer.predict(features)

        print("=" * 60)
        print("Sequence:", feature_file)
        print("True:     ", " | ".join(labels))
        print("Predicted:", " | ".join(prediction))


if __name__ == "__main__":
    main()