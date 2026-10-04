from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def test_required_files_exist():
    required = [
        "index.html",
        "styles.css",
        "app.js",
        "manifest.webmanifest",
        "sw.js",
        "assets/icon.svg",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, f"Missing UI files: {missing}"

def test_manifest():
    manifest = json.loads((ROOT / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["name"].startswith("SignBridge")
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "./"

def test_model_contract_present_in_ui():
    js = (ROOT / "app.js").read_text(encoding="utf-8")
    assert "[T,960]" in js
    assert "speechSynthesis" in js

def test_camera_controls_present():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    for token in ["startBtn", "stopBtn", "resetBtn", "runTestBtn"]:
        assert token in html
