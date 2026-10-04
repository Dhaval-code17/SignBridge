from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "CISLR" / "dataset.csv"
VIDEO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 6


def main() -> None:
    df = pd.read_csv(ANNOTATIONS)

    # Count total examples for every gloss.
    counts = df["gloss"].value_counts()

    # Initial experimental vocabulary:
    # glosses with at least MIN_SAMPLES examples.
    selected_glosses = counts[counts >= MIN_SAMPLES].index.tolist()

    filtered = df[df["gloss"].isin(selected_glosses)].copy()

    # Build absolute video paths.
    filtered["video_path"] = filtered["uid"].astype(str).apply(
        lambda uid: str(VIDEO_ROOT / f"{uid}.mp4")
    )

    # Verify files actually exist.
    filtered["video_exists"] = filtered["video_path"].apply(
        lambda p: Path(p).is_file()
    )

    missing = filtered[~filtered["video_exists"]]

    if not missing.empty:
        print("ERROR: Missing videos:")
        print(missing[["uid", "gloss", "video_path"]].to_string(index=False))
        raise FileNotFoundError(
            f"{len(missing)} video files referenced by the manifest are missing."
        )

    # Keep only the columns required downstream.
    manifest = filtered[
        ["uid", "gloss", "duration", "category", "video_path"]
    ].copy()

    # Sort for reproducibility.
    manifest = manifest.sort_values(
        by=["gloss", "uid"]
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "model1_manifest.csv"
    manifest.to_csv(output_path, index=False)

    print(f"Total selected classes: {len(selected_glosses)}")
    print(f"Total selected videos: {len(manifest)}")
    print(f"Manifest written to: {output_path}")

    print("\nClass distribution:")
    print(
        manifest["gloss"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()