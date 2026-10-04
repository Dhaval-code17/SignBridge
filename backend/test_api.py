import json
import numpy as np
import requests


API_URL = "http://127.0.0.1:8080/recognize"
FEATURE_PATH = "data/continuous/val/seq_000002.npy"


def main():
    features = np.load(FEATURE_PATH)

    payload = {
        "features": features.tolist()
    }

    response = requests.post(
        API_URL,
        json=payload,
    )

    print("Status:", response.status_code)
    print("Response:")
    print(response.json())


if __name__ == "__main__":
    main()