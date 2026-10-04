from pathlib import Path

from preprocessing.video_sampling import load_uniform_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "CISLR"
    / "CISLR_v1.5-a_videos"
    / "CISLR_v1.5-a_videos"
    / "6sQeR9LXn-U_2.mp4"
)


def main() -> None:
    print("Testing:", VIDEO)

    frames = load_uniform_frames(
        VIDEO,
        num_frames=16,
    )

    print("Frames:", frames.shape)
    print("Dtype:", frames.dtype)
    print("Min:", frames.min())
    print("Max:", frames.max())

    assert frames.shape == (16, 300, 300, 3)
    assert frames.dtype.name == "uint8"

    print("Problem video test successful.")


if __name__ == "__main__":
    main()