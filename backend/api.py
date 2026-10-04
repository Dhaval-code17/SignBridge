from fastapi import FastAPI, HTTPException
import numpy as np

from backend.recognizer_service import RecognizerService
from backend.websocket import router as websocket_router


app = FastAPI(
    title="SignBridge Model 2 API",
    description="Temporal sign recognition backend",
    version="1.0.0",
)
app.include_router(websocket_router)

recognizer_service = RecognizerService()


@app.get("/")
def root():
    return {
        "service": "SignBridge Model 2",
        "status": "running",
    }


@app.post("/recognize")
def recognize(payload: dict):
    try:
        features = payload["features"]
        features_array = np.asarray(features, dtype=np.float32)
        result = recognizer_service.recognize(features_array)
        return result
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Missing 'features' field."
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))