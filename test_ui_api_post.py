"""
test_ui_api_post.py
===================
Simulates the exact HTTP POST request sent by the Web UI to http://127.0.0.1:8001/api/recognize_video
"""

import urllib.request
import json
from pathlib import Path

VIDEO_PATH = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level\why are you angry\angry.MP4")
API_URL = "http://127.0.0.1:8001/api/recognize_video"

def main():
    print("="*60)
    print("  Testing Web UI API Endpoint: POST /api/recognize_video")
    print("="*60)

    if not VIDEO_PATH.exists():
        print(f"Error: Video file not found: {VIDEO_PATH}")
        return

    with open(VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    req = urllib.request.Request(
        API_URL,
        data=video_bytes,
        headers={
            "Content-Type": "video/mp4",
            "Content-Length": str(len(video_bytes))
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)
            print("\n--- UI API RESPONSE RECEIVED ---")
            print("HTTP Status Code:", response.getcode())
            print("Success (ok)    :", data.get("ok"))
            print("Predicted Gloss :", data.get("predicted_gloss"))
            print("Similarity Score:", data.get("similarity"))
            print("\nTop 5 Recognized Sentences:")
            for i, item in enumerate(data.get("top5", []), 1):
                marker = " <<< MATCHES UI!" if "angry" in item["gloss"] else ""
                print(f"  {i}. {item['gloss']} (Similarity: {item['similarity']:.4f}){marker}")
    except Exception as e:
        print("API Post Error:", e)

if __name__ == "__main__":
    main()
