from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from backend.recognizer_service import RecognizerService


router = APIRouter()

recognizer_service = RecognizerService()

WINDOW_SIZE = 64


@router.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()

    feature_buffer = deque(maxlen=WINDOW_SIZE)

    try:
        while True:
            data = await websocket.receive_json()

            features = np.asarray(
                data["features"],
                dtype=np.float32
            )

            timestamp = data.get("timestamp")

            if timestamp is None:
                await websocket.send_json({
                    "error": "Missing 'timestamp' field."
                })
                continue

            if features.shape != (960,):
                await websocket.send_json({
                    "error": "Each frame must contain exactly 960 features."
                })
                continue

            feature_buffer.append(features)

            sequence_features = np.stack(feature_buffer)

            result = recognizer_service.recognize(
                sequence_features
            )

            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")