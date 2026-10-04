import numpy as np
from inference.continuous_recognizer import ContinuousRecognizer


class RecognizerService:
    def __init__(self):
        self.recognizer = ContinuousRecognizer()

    def recognize(self, features):
        """
        Convert Person 1's 960-dim feature vector(s)
        into a sequence of recognized signs.
        Supports both single frame 1D [960] and temporal sequence 2D [T, 960].
        """
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)

        if features.ndim == 1:
            if features.shape[0] != 960:
                raise ValueError(f"1D feature array must have dimension 960, got {features.shape[0]}")
            features = features[np.newaxis, :]  # reshape to [1, 960]

        elif features.ndim == 2:
            if features.shape[1] != 960:
                raise ValueError(f"2D feature matrix must have shape [T, 960], got {features.shape}")
        else:
            raise ValueError(f"Features must be 1D [960] or 2D [T, 960], got ndim={features.ndim}")

        signs = self.recognizer.predict(features)
        sequence = " ".join(signs)

        return {
            "sequence": sequence
        }