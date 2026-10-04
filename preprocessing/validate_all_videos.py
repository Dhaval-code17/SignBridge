from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from preprocessing.video_sampling import load_uniform_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model1_manifest.csv"
)


def main() -> None:
    df = pd.read_csv(MANIFEST)

    failed: list[dict[str, str]] = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Validating videos",
    ):
        try:
            frames = load_uniform_frames(
                row.video_path,
                num_frames=16,
            )

            if frames.shape != (16, 300, 300, 3):
                failed.append(
                    {
                        "uid": row.uid,
                        "gloss": row.gloss,
                        "reason": (
                            f"Unexpected shape: {frames.shape}"
                        ),
                    }
                )

        except Exception as exc:
            failed.append(
                {
                    "uid": row.uid,
                    "gloss": row.gloss,
                    "reason": str(exc),
                }
            )

    print()
    print("=" * 60)
    print("VIDEO VALIDATION")
    print("=" * 60)
    print("Total videos:", len(df))
    print("Failed videos:", len(failed))

    if failed:
        print("\nFailures:")
        for item in failed:
            print(
                f"{item['uid']} | "
                f"{item['gloss']} | "
                f"{item['reason']}"
            )

        raise RuntimeError(
            f"{len(failed)} videos failed validation."
        )

    print("All videos successfully decoded.")
    print("Every video produced 16 frames.")
    print("=" * 60)


if __name__ == "__main__":
    main()