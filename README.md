# SignBridge — Model 1

> **Real-time Indian Sign Language (ISL) recognition powered by MobileNetV3 + Contrastive Learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

SignBridge Model 1 is an end-to-end sign language recognition system that:

- Extracts **960-dim feature embeddings** from video frames using a fine-tuned MobileNetV3-Large backbone.
- Performs **one-shot / few-shot recognition** via nearest-neighbour matching against a pre-built reference library of 5,686 training clips spanning 4,764 ISL glosses.
- Serves predictions through a **local HTTP server** with a built-in Progressive Web App (PWA) UI that supports both live camera inference and uploaded video recognition.

---

## Project Structure

```
signbridge_model1/
├── signbridge_server.py        # Main server entry point (run this)
├── inference.py                # Standalone inference helper
├── requirements.txt            # Pinned Python dependencies
├── setup.py                    # Optional installable package
│
├── configs/                    # JSON configs & class mappings
│   ├── classes.json
│   ├── classes_final.json      # ⚠️  gitignored (large)
│   └── feature_extractor_config.json
│
├── models/                     # Model definitions & artefacts
│   ├── __init__.py
│   ├── mobilenet_model.py      # MobileNetV3-Large encoder
│   ├── cnn_baseline.py         # CNN baseline (reference)
│   ├── checkpoints/            # ⚠️  gitignored — download separately
│   ├── onnx/                   # ⚠️  gitignored — ONNX exports
│   ├── features/               # Pre-computed reference features
│   └── evaluation/             # Saved eval metrics & confusion matrices
│
├── training/                   # All training scripts
│   ├── dataset.py
│   ├── feature_extractor.py
│   ├── train_final_contrastive.py   # ✅ Primary training script
│   ├── train_final_mobilenet.py
│   ├── train_mobilenet_stage2.py
│   ├── train_mobilenet.py
│   ├── train_cnn.py
│   ├── evaluate_contrastive_encoder.py
│   ├── evaluate_one_shot_mobilenet.py
│   ├── evaluate_stage2_temporal_matching.py
│   ├── evaluate_temporal_matching.py
│   ├── export_mobilenet_encoder_onnx.py
│   ├── extract_continuous_features.py
│   ├── create_prototype_continuous_39.py
│   ├── final_model1_handoff.py      # Full training pipeline summary
│   ├── verify_onnx_real_video.py
│   └── preprocessing/              # Dataset split scripts
│
├── preprocessing/              # Video transforms & manifest utilities
│   ├── transforms.py
│   ├── video_sampling.py
│   └── ...
│
├── data/                       # ⚠️  gitignored — dataset lives here
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
├── inference/                  # Inference utilities (importable module)
├── evaluation/                 # Standalone evaluation scripts
├── features/                   # Extracted feature caches (gitignored)
├── logs/                       # Training logs (gitignored)
│
└── ui/                         # Frontend PWA
    ├── index.html
    ├── app.js                  # Main application logic
    ├── ui_app.js               # UI helpers
    ├── styles.css
    ├── sw.js                   # Service Worker
    ├── manifest.webmanifest
    └── assets/
```

---

## Quick Start

### 1 — Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| CUDA (optional, recommended) | ≥ 12.0 |
| GPU VRAM | ≥ 6 GB (for training) |

### 2 — Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install requirements
pip install -r requirements.txt
```

### 3 — Download Model Checkpoints

The `.pth` and `.onnx` files are **not included in this repo** (too large for Git).
Download them and place them at the paths shown below:

```
models/
└── checkpoints/
    ├── mobilenet_v3_full_vocab_contrastive_best.pth  ← primary model
    ├── mobilenet_v3_full_vocab_best.pth
    ├── mobilenet_v3_stage1_best.pth
    ├── mobilenet_v3_stage2_best.pth
    └── cnn_baseline_best.pth
```

> **Tip:** Use [Git LFS](https://git-lfs.com/) or a GitHub Release to distribute these files.

### 4 — Prepare Reference Library

The server loads a pre-built feature matrix from `data/processed/final_model1_features_train/`.
Run the feature extraction script once after placing your dataset:

```bash
python training/extract_continuous_features.py
```

### 5 — Run the Server

```bash
python signbridge_server.py
```

Then open **http://localhost:8001** in your browser.

---

## Training

To retrain from scratch using the contrastive encoder approach:

```bash
python training/train_final_contrastive.py
```

See [`training/final_model1_handoff.py`](training/final_model1_handoff.py) for the complete pipeline walkthrough.

---

## Model Architecture

```
Input Video (T frames × 224×224)
        │
        ▼
MobileNetV3-Large (ImageNet pretrained)
  [features layer — pool5 output]
        │
        ▼
960-dim L2-normalised embedding per frame
        │
        ▼
Temporal aggregation (mean pool over T frames)
        │
        ▼
960-dim video-level descriptor
        │
        ▼
Cosine nearest-neighbour search
against reference library (5686 clips × 960)
        │
        ▼
Top-K gloss predictions + confidence scores
```

---

## API Reference

The server exposes a minimal REST API:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server health check |
| `/api/model1` | POST | Recognise sign from video frames |

### POST `/api/model1`

**Request body (JSON)**

```json
{
  "frames": ["<base64-encoded-JPEG>", "..."],
  "source": "camera"
}
```

**Response**

```json
{
  "predictions": [
    { "gloss": "HELLO", "score": 0.94 },
    { "gloss": "HI",    "score": 0.71 }
  ],
  "inference_ms": 42
}
```

---

## Evaluation Results

| Metric | Value |
|---|---|
| Reference library size | 5,686 clips |
| Vocabulary | 4,764 glosses |
| Embedding dimension | 960 |
| Device | CUDA (GPU) |

Full evaluation reports are in `models/evaluation/` (gitignored from the repo; run locally).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [CISLR Dataset](https://www.cislr.org/) — ISL video corpus
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — Howard et al., 2019
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) — Khosla et al., 2020
