import sys
sys.path.insert(0, '.')
import numpy as np
from fastapi.testclient import TestClient
from backend.api import app

def test_api():
    print("Testing FastAPI HTTP endpoint...")
    client = TestClient(app)
    
    # 1. Root GET check
    res = client.get("/")
    assert res.status_code == 200
    print("Root GET response:", res.json())
    
    # 2. Recognize POST check with 1D single frame feature vector [960]
    single_frame_features = np.random.randn(960).tolist()
    payload_1d = {
        "features": single_frame_features,
        "timestamp": 12.34
    }
    res_1d = client.post("/recognize", json=payload_1d)
    print("Recognize POST (1D features) status:", res_1d.status_code)
    print("Recognize POST (1D features) output:", res_1d.json())
    assert res_1d.status_code == 200
    assert "sequence" in res_1d.json()

    # 3. Recognize POST check with 2D temporal sequence matrix [T=20, 960]
    sequence_features = np.random.randn(20, 960).tolist()
    payload_2d = {
        "features": sequence_features,
        "timestamp": 12.34
    }
    res_2d = client.post("/recognize", json=payload_2d)
    print("Recognize POST (2D sequence) status:", res_2d.status_code)
    print("Recognize POST (2D sequence) output:", res_2d.json())
    assert res_2d.status_code == 200
    assert "sequence" in res_2d.json()

    print("All FastAPI backend tests passed successfully!")

if __name__ == "__main__":
    test_api()
