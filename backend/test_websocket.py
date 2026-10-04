import asyncio
import json

import numpy as np
import websockets


WS_URL = "ws://127.0.0.1:8080/ws/recognize"
FEATURE_PATH = "data/continuous/val/seq_000002.npy"


async def main():
    features = np.load(FEATURE_PATH)

    async with websockets.connect(WS_URL) as websocket:

        for frame_index, frame in enumerate(features):
            payload = {
                "features": frame.tolist(),
                "timestamp": frame_index / 8.0
            }

            await websocket.send(json.dumps(payload))

            response = await websocket.recv()

            print(response)


if __name__ == "__main__":
    asyncio.run(main())