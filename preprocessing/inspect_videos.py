from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "model1_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "video_properties.csv"


def inspect_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "actual_duration": None,
            "readable": False,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if fps and fps > 0 and frame_count >= 0:
        actual_duration = frame_count / fps
    else:
        actual_duration = None

    cap.release()

    return {
        "fps": fps,
        "frame_count": int(frame_count) if frame_count >= 0 else None,
        "width": int(width) if width > 0 else None,
        "height": int(height) if height > 0 else None,
        "actual_duration": actual_duration,
        "readable": True,
    }


def main() -> None:
    df = pd.read_csv(MANIFEST_PATH)

    # Inspect every video in the current experiment.
    properties = df["video_path"].apply(inspect_video).apply(pd.Series)

    result = pd.concat(
        [
            df[["uid", "gloss", "duration", "video_path"]],
            properties,
        ],
        axis=1,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Videos inspected: {len(result)}")
    print(f"Unreadable videos: {(~result['readable']).sum()}")

    print("\nFPS:")
    print(result["fps"].describe().to_string())

    print("\nFrame count:")
    print(result["frame_count"].describe().to_string())

    print("\nWidth:")
    print(result["width"].value_counts().head(10).to_string())

    print("\nHeight:")
    print(result["height"].value_counts().head(10).to_string())

    print("\nDuration:")
    print(result["actual_duration"].describe().to_string())

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()