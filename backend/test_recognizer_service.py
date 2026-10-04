import numpy as np

from backend.recognizer_service import RecognizerService


def main():
    service = RecognizerService()

    features = np.load(
        "data/continuous/val/seq_000002.npy"
    )

    result = service.recognize(features)

    print("Backend result:")
    print(result)


if __name__ == "__main__":
    main()