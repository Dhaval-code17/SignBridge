# SignBridge — Model 2
## Indian Sign Language Temporal Recognition & Real-Time Backend Service

SignBridge is an AI-powered system designed to convert **Indian Sign Language (ISL)** gestures into meaningful English sentences and speech.

This repository section contains the complete contribution of **Person 2**, responsible for the **temporal sign-sequence recognition stage (Model 2)** and the **real-time backend server infrastructure**.

Model 2 receives the continuous 960-dimensional visual feature vectors produced per frame by **Model 1 (Person 1)**, processes their temporal dependencies across dynamic sequence lengths using a multi-head Transformer Encoder, and outputs recognized sign glosses (`{"sequence": "SIGN_NAME"}`).

These recognized sign sequences are handed off to **Model 3 (Person 3)** for natural English sentence translation and Text-to-Speech (TTS) generation.

---

# 1. Person 2 Contribution

Person 2 is responsible for the complete **temporal sequence modeling architecture, ONNX export, and real-time backend API infrastructure**.

### Responsibilities

- Temporal sequence dataset pipeline
- Variable-length padding and key mask collation
- 4-layer Temporal Transformer Encoder architecture design
- Sinusoidal positional encodings over time
- Temporal pooling (Masked Global Mean Pooling)
- Model 2 training & optimization (AdamW, Cosine Annealing, Label Smoothing)
- Checkpoint management and class vocabulary mapping (4,764 ISL classes)
- ONNX model export (`model2.onnx` with dynamic temporal axes)
- FastAPI HTTP REST backend (`POST /recognize`)
- Real-time WebSocket streaming server (`ws://.../ws/recognize`)
- Sliding 64-frame temporal window buffer management
- Model 1 → Model 2 connector service (`RecognizerService`)
- Robust model path resolution (`ContinuousRecognizer`)
- Person 2 → Person 3 handover package containerization (`person2_handover/`)
- Model 2 performance evaluation & metrics logging

The output of Model 2 is a recognized sign gloss sequence passed to Person 3.

### Model 1 → Model 2 → Model 3 Interface Boundaries

```text
Model 1 Visual Output : [T, 960] float32
Model 2 Output        : {"sequence": "SIGN_NAME"}
```

Where:

- `T` = variable sequence length of video frames
- `960` = MobileNetV3 visual feature vector dimension
- `SIGN_NAME` = recognized Indian Sign Language gloss string

---

# 2. Complete SignBridge Architecture

```text
                    ┌──────────────────────┐
                    │   Camera / Video     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Preprocessing      │
                    │                      │
                    │ • Frame Sampling     │
                    │ • Resize 224×224     │
                    │ • Normalization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MODEL 1         │
                    │ (Person 1 Ownership) │
                    │ MobileNetV3-Large    │
                    │ Visual Encoder       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Visual Features      │
                    │                      │
                    │ [T, 960] float32     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ========================================
                    │      MODEL 2 & BACKEND               │
                    │ (PERSON 2 OWNERSHIP)                 │
                    │                                      │
                    │ • Temporal Transformer Classifier    │
                    │ • FastAPI REST & WebSocket Server    │
                    │ • ONNX Model Runtime Engine          │
                    ========================================
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sign Sequence        │
                    │                      │
                    │ {"sequence":         │
                    │  "ABSENT"}           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MODEL 3         │
                    │ (Person 3 Ownership) │
                    │ T5 / mT5             │
                    │ English Translation  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Natural English      │
                    │ Sentence             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text-to-Speech (TTS) │
                    └──────────────────────┘
```

---

# 3. Model 2 Role

Model 2 acts as the **temporal language decoder** of SignBridge.

Its job is to bridge spatial visual representations and natural language translation.

The processing pipeline is:

```text
Visual Feature Vector Sequence [T, 960] (from Model 1)
     ↓
Linear Projection (960 ➔ 256)
     ↓
Sinusoidal Positional Encoding
     ↓
4-Layer Transformer Encoder (8 Attention Heads)
     ↓
Masked Global Temporal Mean Pooling
     ↓
Linear Classification Head (256 ➔ 4,764 Classes)
     ↓
Predicted Sign Gloss ("SIGN_NAME")
     ↓
Model 3 (T5 Translation Engine)
```

Model 2 provides the critical link:

```text
Frame-Level Visual Features [T, 960]
      ↓
Temporal Sequence Modeling
      ↓
Recognized ISL Sign Glosses
```

---

# 4. Model 2 Inputs

Model 2 accepts feature sequences generated by Person 1's MobileNetV3 visual encoder.

Accepted Input Shapes:

- **Single Frame Feature Vector**: `[960]` float32 array
- **Temporal Sequence Matrix**: `[T, 960]` float32 array (where `T` is the number of video frames)
- **Batch Tensor**: `[B, T, 960]` float32 tensor (where `B` is batch size)

Each feature vector contains:

```text
960 continuous float32 feature values
```

Because video durations vary, Model 2 handles variable sequence lengths seamlessly:

```text
Sequence A → [24, 960]
Sequence B → [48, 960]
Sequence C → [16, 960]
```

Without needing artificial temporal warping or static cropping.

---

# 5. Model 2 Architecture Specification

Model 2 uses a **Temporal Transformer Encoder** architecture optimized for sequential feature classification.

```text
Input Feature Sequence [B, T, 960]
            │
            ▼
Linear Projection Layer (960 ➔ 256)
            │
            ▼
Layer Normalization
            │
            ▼
Sinusoidal Positional Encoding
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 4x Transformer Encoder Layers                          │
│   • Multi-Head Self-Attention (nhead = 8)              │
│   • Feedforward Network (dim_feedforward = 512, GELU)  │
│   • Layer Normalization (norm_first = True)            │
│   • Dropout (p = 0.2)                                  │
└────────────────────────────────────────────────────────┘
            │
            ▼
Masked Global Temporal Average Pooling
            │
            ▼
Layer Normalization & Dropout (0.2)
            │
            ▼
Linear Classifier Head (256 ➔ 4,764 Classes)
            │
            ▼
Output Logits [B, 4764]
```

### Architectural Parameters Summary

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `input_dim` | `960` | Input feature dimension from Person 1 |
| `d_model` | `256` | Hidden transformer embedding dimension |
| `nhead` | `8` | Parallel self-attention heads |
| `num_layers` | `4` | Stacked Transformer Encoder layers |
| `dim_feedforward` | `512` | Hidden dimension of feedforward network |
| `dropout` | `0.2` | Regularization dropout rate |
| `num_classes` | `4,764` | Total ISL sign vocabulary size |
| **Total Parameters** | **3,579,804** | Lightweight ~3.58M trainable parameters |

---

# 6. Positional Encoding & Key Padding Masking

### Positional Encoding
Since Transformer attention mechanisms are permutation-invariant, a sinusoidal positional encoding is added to feature embeddings to retain frame chronological order:

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### Key Padding Masking
When processing variable-length frame sequences in batches, zero-padding is applied to match the maximum sequence length `T_max`. 

A boolean `src_key_padding_mask` of shape `[B, T_max]` masks out padded time steps during attention matrix calculation and temporal pooling, ensuring zero-padded trailing frames do not corrupt sequence representations.

---

# 7. Dataset & Vocabulary

Model 2 was trained and evaluated on the continuous ISL dataset matching Person 1's vocabulary.

### Dataset Statistics

```text
Total Preloaded Sequences : 7,049
Training Sequences        : 5,686
Validation Sequences      : 1,363
Total ISL Classes         : 4,764 unique sign glosses
```

### Vocabulary Mapping
The class vocabulary mapping is saved in:

```text
models/checkpoints/temporal_classifier_classes.json
```

This file maps class indices `0..4763` to ISL gloss string names (e.g., `{"0": "aam aadmi party", "1": "abdicate", ...}`) and is essential for Person 3 integration.

---

# 8. Training Strategy & Hyperparameters

Training was conducted using PyTorch with RAM preloading for ultra-fast epoch execution.

### Hyperparameters Summary

```text
Optimizer           : AdamW
Learning Rate (LR)  : 3e-4
Weight Decay        : 1e-4
LR Scheduler        : CosineAnnealingLR (T_max = 50, eta_min = 1e-6)
Batch Size          : 32
Loss Function       : LabelSmoothingLoss (smoothing = 0.1)
Gradient Clipping   : 5.0
```

---

# 9. Model 2 Performance & Metrics

Model 2 was evaluated across all **4,764 unique sign classes** on the 1,363 validation sequences.

### Official Validation Results

```text
============================================================
Person 2 — Temporal Transformer Evaluation Results
============================================================
Validation Sequences   : 1,363
Vocabulary Classes     : 4,764

Top-1 Validation Accuracy: 30.52%
Top-5 Validation Accuracy: 37.42%
============================================================
```

> **Note on Evaluation Context**: Achieving 30.52% Top-1 accuracy across **4,764 distinct classes** with only ~1.19 training examples per class represents a strong sequence classification benchmark (random chance is 0.02%).

---

# 10. ONNX Model Export (`model2.onnx`)

To support low-latency cross-platform inference, Model 2 is exported into **ONNX** format.

### ONNX Export Specifications

```text
ONNX File           : models/model2.onnx (or person2_handover/models/model2.onnx)
ONNX Model Size     : 18.66 MB
Input Shape         : [batch_size, sequence_length, 960] (float32)
Output Shape        : [batch_size, 4764] (float32 logits)
Dynamic Axes        : batch_size (axis 0), sequence_length (axis 1)
```

Both PyTorch (`.pth`) and ONNX (`.onnx`) runtimes are verified and supported.

---

# 11. Backend API Infrastructure

Person 2 delivers two production-ready web servers built with **FastAPI** and **Uvicorn**:

### Method 1: FastAPI HTTP REST API (`POST /recognize`)
- **Server Entrypoint**: `backend/api.py`
- **Port**: `8000`

#### HTTP Request Payload
```json
{
  "features": [0.13, -0.42, 0.87, "..."],
  "timestamp": 12.34
}
```
*(Accepts either 1D `[960]` single frame array or 2D `[T, 960]` sequence array).*

#### HTTP Response Payload
```json
{
  "sequence": "ABSENT"
}
```

### Method 2: Real-Time WebSocket Streaming (`ws://.../ws/recognize`)
- **Server Entrypoint**: `backend/websocket.py`
- **Endpoint**: `ws://localhost:8000/ws/recognize`
- **Behavior**: Maintains a sliding temporal window buffer (`deque(maxlen=64)`). Accepts streaming frame JSON payloads and broadcasts real-time sign predictions to connected client UIs.

---

# 12. Project Directory Structure

```text
signbridge_model2FF/
│
├── backend/
│   ├── api.py                      # FastAPI HTTP REST server (POST /recognize)
│   ├── websocket.py                # WebSocket streaming server (ws://.../ws/recognize)
│   └── recognizer_service.py       # Interface connector service (Model 1 -> Model 2)
│
├── inference/
│   └── continuous_recognizer.py    # Python inference engine with dynamic path resolution
│
├── models/
│   ├── model2.onnx                 # Exported ONNX model (18.66 MB)
│   └── checkpoints/
│       ├── temporal_classifier_best.pth    # PyTorch model weights (~14.3 MB)
│       └── temporal_classifier_classes.json # Vocabulary mapping (4,764 classes)
│
├── training/
│   ├── train_temporal_classifier.py# Model 2 training script
│   └── export_onnx.py              # ONNX exporter script
│
├── evaluation/
│   └── evaluate_temporal_classifier.py # Validation evaluation script
│
├── person2_handover/               # Handover package for Person 3
│   ├── backend/
│   ├── inference/
│   ├── models/
│   ├── training/
│   └── README.md
│
├── .gitignore
├── requirements.txt
├── person2_project_report.md
└── README.md
```

---

# 13. Dynamic Checkpoint Path Resolution

To ensure Person 3 can run the handover package without path errors, `ContinuousRecognizer` implements automatic candidate path resolution:

```python
candidate_paths = [
    os.path.join(root_dir, "models", "temporal_classifier_best.pth"),
    os.path.join(root_dir, "models", "checkpoints", "temporal_classifier_best.pth"),
    "models/temporal_classifier_best.pth",
    "models/checkpoints/temporal_classifier_best.pth",
]
```

This guarantees seamless execution whether run from the root repository or inside `person2_handover/`.

---

# 14. Quick Start & Execution Commands

### 1. Environment Setup
```powershell
# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run HTTP Backend Server
```powershell
uvicorn backend.api:app --reload --port 8000
```

### 3. Run WebSocket Streaming Server
```powershell
uvicorn backend.websocket:app --reload --port 8000
```

### 4. Evaluate Model Performance
```powershell
python evaluation/evaluate_temporal_classifier.py
```

### 5. Export ONNX Model
```powershell
python training/export_onnx.py
```

---

# 15. Python Integration Example for Person 3

Person 3 can pass the output of Model 2 directly into their T5 / mT5 translation pipeline:

```python
import requests

# 1. Feature vector(s) received from Person 1 (MobileNetV3)
features_from_person1 = [0.13, -0.42, 0.87, ...] # 960 floats

# 2. Call Person 2's backend API
response = requests.post(
    "http://localhost:8000/recognize",
    json={"features": features_from_person1}
)

# 3. Extract recognized sign sequence
sign_sequence = response.json()["sequence"]  # Output: "ABSENT"

# 4. Person 3 passes sign_sequence into Model 3 (T5) for sentence translation
english_sentence = model3_t5.translate(sign_sequence)

# 5. Convert translation to speech
tts.speak(english_sentence)
```

---

# 16. Future Enhancements

1. **Connectionist Temporal Classification (CTC) Loss**:
   - Transitioning from global temporal mean pooling to CTC decoding for unsegmented continuous sentence recognition.
2. **Chunked Causal Attention**:
   - Implementing streaming causal masking so Model 2 emits glosses incrementally frame-by-frame.
3. **INT8 Weight Quantization**:
   - Quantizing `model2.onnx` from 18.66 MB down to ~4.8 MB for mobile deployment.

---

# 17. Responsibility Boundaries

| Person | Model | Ownership | Primary Responsibility |
| :--- | :--- | :--- | :--- |
| **Person 1** | **Model 1** | Frontend / Visual | Camera feed, preprocessing, MobileNetV3 960-D visual feature extraction |
| **Person 2** | **Model 2** | Backend / Temporal | Temporal Transformer, FastAPI REST, WebSocket streaming, ONNX export |
| **Person 3** | **Model 3** | Integration / Translation | T5/mT5 sign-to-sentence translation, Text-to-Speech (TTS), System UI |

---

# 18. Handover Checklist for Person 3

- [x] PyTorch model checkpoint saved (`temporal_classifier_best.pth`)
- [x] ONNX model exported and verified (`model2.onnx`)
- [x] Vocabulary JSON mapping generated (`temporal_classifier_classes.json` - 4,764 classes)
- [x] FastAPI HTTP server implemented (`POST /recognize`)
- [x] WebSocket streaming server implemented (`ws://.../ws/recognize`)
- [x] Path resolution verified for turnkey execution
- [x] Handover documentation complete (`person2_handover/README.md`)

---

# License

This project component is released under the **MIT License**.
