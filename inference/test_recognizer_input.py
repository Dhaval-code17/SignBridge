import numpy as np
import torch

from inference.continuous_recognizer import ContinuousRecognizer


def main():
    recognizer = ContinuousRecognizer()

    features = np.load(
        "data/continuous/val/seq_000002.npy"
    )

    print("Original shape:", features.shape)

    # Test NumPy input
    numpy_prediction = recognizer.predict(features)

    print()
    print("NumPy input:")
    print(numpy_prediction)

    # Test PyTorch input
    tensor_features = torch.tensor(
        features,
        dtype=torch.float32
    )

    tensor_prediction = recognizer.predict(
        tensor_features
    )

    print()
    print("PyTorch input:")
    print(tensor_prediction)


if __name__ == "__main__":
    main()