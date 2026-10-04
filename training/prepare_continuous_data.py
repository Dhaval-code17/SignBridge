import os
import json
import zipfile
import io

import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

RE_ZIP = r"C:\Users\nihav\Downloads\re.zip"
FEATURES_ZIP = r"C:\Users\nihav\Downloads\features.zip"

OUTPUT_DIR = "data/continuous"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

print("Loading metadata...")

with zipfile.ZipFile(RE_ZIP, "r") as z:
    metadata = json.loads(
        z.read("metadata.json")
    )


sequences = metadata["sequences"]

print("Number of sequences:", len(sequences))


# --------------------------------------------------
# Open feature ZIP
# --------------------------------------------------

print("Opening features ZIP...")

features_zip = zipfile.ZipFile(
    FEATURES_ZIP,
    "r"
)


# --------------------------------------------------
# Prepare each continuous sequence
# --------------------------------------------------

for index, sequence in enumerate(sequences, start=1):

    sequence_id = sequence["sequence_id"]
    source_uids = sequence["source_uids"]
    labels = sequence["labels"]

    print(
        f"[{index}/{len(sequences)}] "
        f"{sequence_id}: {labels}"
    )

    feature_parts = []

    for uid in source_uids:

        feature_path = f"features/{uid}.npy"

        if feature_path not in features_zip.namelist():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        feature_bytes = features_zip.read(
            feature_path
        )

        features = np.load(
            io.BytesIO(feature_bytes)
        )

        if features.ndim != 2:
            raise ValueError(
                f"{feature_path} has shape "
                f"{features.shape}, expected [T, 960]"
            )

        if features.shape[1] != 960:
            raise ValueError(
                f"{feature_path} has feature dimension "
                f"{features.shape[1]}, expected 960"
            )

        feature_parts.append(features)

    # Concatenate signs in their original order
    combined_features = np.concatenate(
        feature_parts,
        axis=0
    )

    # Save feature sequence
    output_feature_path = os.path.join(
        OUTPUT_DIR,
        f"{sequence_id}.npy"
    )

    np.save(
        output_feature_path,
        combined_features
    )

    # Save labels
    output_label_path = os.path.join(
        OUTPUT_DIR,
        f"{sequence_id}.json"
    )

    with open(
        output_label_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            labels,
            f,
            indent=2,
            ensure_ascii=False
        )


# --------------------------------------------------
# Finish
# --------------------------------------------------

features_zip.close()

print()
print("======================================")
print("Preparation complete!")
print("Output directory:", OUTPUT_DIR)
print("Sequences created:", len(sequences))
print("======================================")