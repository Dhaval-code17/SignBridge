from __future__ import annotations

import base64
import json
import re
import tempfile
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# SIGNBRIDGE REAL MODEL 1 + VIDEO RECOGNITION SERVER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
UI_ROOT = PROJECT_ROOT / "ui"

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_full_vocab_contrastive_best.pth"
)

TRAIN_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_model1_features_train"
)

TRAIN_MANIFEST = TRAIN_FEATURE_DIR / "sequences.csv"

HOST = "127.0.0.1"
PORT = 8001

FEATURE_DIM = 960
IMAGE_SIZE = 224

MODEL1_MAX_FRAMES = 32

UPLOAD_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
UPLOAD_SAMPLE_FPS = 8.0
MAX_VIDEO_SECONDS = 15.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL 1
# ============================================================

class Model1Encoder(nn.Module):
    """
    SignBridge Model 1.

    MobileNetV3-Large backbone.

    Output:
        [B, 960]

    The 128-D projection head is NOT used.
    Classifier logits are NOT used.
    """

    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )


        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        return torch.flatten(x, 1)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_model():
    if not CHECKPOINT.exists():
        print("=" * 70)
        print("WARNING: Model 1 checkpoint not found.")
        print(f"Path: {CHECKPOINT}")
        print("Running Model 1 with default initialized MobileNetV3 weights.")
        print("=" * 70)
        model = Model1Encoder()
        model.to(DEVICE)
        model.eval()
        return model


    print("=" * 70)
    print("Loading Model 1 checkpoint...")
    print(f"Checkpoint: {CHECKPOINT}")
    print("=" * 70)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = Model1Encoder()

    expected_state = model.state_dict()
    expected_keys = set(expected_state.keys())

    # --------------------------------------------------------
    # Locate state dictionary
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "backbone_state_dict" in checkpoint
    ):
        raw_state = checkpoint["backbone_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        raw_state = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict):

        raw_state = None

        for value in checkpoint.values():

            if not isinstance(value, dict):
                continue

            tensor_count = sum(
                1
                for v in value.values()
                if torch.is_tensor(v)
            )

            if tensor_count >= 100:
                raw_state = value
                break

        if raw_state is None:
            raise ValueError(
                "Could not find tensor state dictionary."
            )

    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

    print(
        f"Checkpoint tensor parameters: "
        f"{len(raw_state)}"
    )

    # --------------------------------------------------------
    # Normalize checkpoint keys
    # --------------------------------------------------------

    normalized = {}

    for key, value in raw_state.items():

        if not torch.is_tensor(value):
            continue

        key = str(key)

        prefixes = [
            "module.",
            "model.",
            "encoder.",
            "backbone.",
        ]

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if key.startswith(prefix):

                    key = key[len(prefix):]
                    changed = True

                    break

        normalized[key] = value

    # --------------------------------------------------------
    # Map checkpoint keys to MobileNet keys
    #
    # Checkpoint:
    #     0.0.weight
    #
    # Model:
    #     features.0.0.weight
    # --------------------------------------------------------

    remapped = {}

    for expected_key in expected_keys:

        if not expected_key.startswith("features."):
            continue

        target = expected_key[
            len("features.") :
        ]

        if target in normalized:

            remapped[expected_key] = normalized[target]

    print(
        f"Mapped MobileNet parameters: "
        f"{len(remapped)}/{len(expected_keys)}"
    )

    # --------------------------------------------------------
    # Validate complete mapping
    # --------------------------------------------------------

    missing = [
        key
        for key in expected_keys
        if key not in remapped
    ]

    if missing:

        print("\nMissing parameters:")

        for key in missing[:30]:
            print(" ", key)

        raise RuntimeError(
            f"Model 1 checkpoint mapping incomplete. "
            f"Missing {len(missing)} parameters."
        )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    for key, value in remapped.items():

        if value.shape != expected_state[key].shape:

            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)} "
                f"expected={tuple(expected_state[key].shape)}"
            )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        remapped,
        strict=False,
    )

    model.eval()
    model.to(DEVICE)

    # --------------------------------------------------------
    # CUDA / CPU warmup
    # --------------------------------------------------------

    dummy = torch.zeros(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    with torch.inference_mode():
        _ = model(dummy)

    print("=" * 70)
    print("MODEL 1 LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"Device:        {DEVICE}")
    print(f"Loaded params: {len(remapped)}")
    print("Output:        [B,960]")
    print("Projection:    NOT USED")
    print("Classifier:    NOT USED")
    print("=" * 70)

    return model


MODEL = load_model()


# ============================================================
# MODEL 3 (T5 TRANSLATION)
# ============================================================

MODEL3_DIR = PROJECT_ROOT / "model3" / "SignBridge_model" / "model3_final"
MODEL3_TOKENIZER = None
MODEL3_MODEL = None


def load_model3():
    global MODEL3_TOKENIZER, MODEL3_MODEL
    if not MODEL3_DIR.exists():
        print("=" * 70)
        print("WARNING: Model 3 directory not found.")
        print(f"Path: {MODEL3_DIR}")
        print("=" * 70)
        return

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        print("=" * 70)
        print("Loading Model 3 checkpoint...")
        print(f"Checkpoint: {MODEL3_DIR}")
        print("=" * 70)

        MODEL3_TOKENIZER = T5Tokenizer.from_pretrained(str(MODEL3_DIR))
        MODEL3_MODEL = T5ForConditionalGeneration.from_pretrained(str(MODEL3_DIR))
        MODEL3_MODEL.to(DEVICE)
        MODEL3_MODEL.eval()

        print("=" * 70)
        print("MODEL 3 LOADED SUCCESSFULLY")
        print("=" * 70)
    except Exception as exc:
        print(f"WARNING: Failed to load Model 3: {exc}")


def translate_sequence(gloss_text: str) -> str:
    global MODEL3_TOKENIZER, MODEL3_MODEL
    if MODEL3_MODEL is None or MODEL3_TOKENIZER is None:
        return gloss_text

    try:
        gloss_text = gloss_text.strip()
        if not gloss_text:
            return ""

        prompt = f"translate Sign to English: {gloss_text}"
        inputs = MODEL3_TOKENIZER(
            prompt, return_tensors="pt", max_length=128, truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = MODEL3_MODEL.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        translated = MODEL3_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        return translated
    except Exception as exc:
        print(f"Model 3 translation error: {exc}")
        return gloss_text


load_model3()



# ============================================================
# PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# REFERENCE LIBRARY
# ============================================================

REFERENCE_MATRIX = None
REFERENCE_GLOSSES = None
GLOSS_TO_INDICES = None


def l2_normalize(vector):
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return vector

    return vector / norm


def build_reference_library():

    global REFERENCE_MATRIX
    global REFERENCE_GLOSSES
    global GLOSS_TO_INDICES

    proto_file = PROJECT_ROOT / "models" / "checkpoints" / "prototype_matcher.npz"
    if proto_file.exists():
        print("=" * 70)
        print("Loading Sentence Prototype Matcher for ISL Model 2...")
        print(f"Path: {proto_file}")
        pdata = np.load(proto_file)
        REFERENCE_MATRIX = pdata["prototypes"]
        labels = list(pdata["labels"])
        REFERENCE_GLOSSES = labels
        GLOSS_TO_INDICES = {lbl: [i] for i, lbl in enumerate(labels)}
        print(f"Loaded {len(labels)} sentence prototypes! Prototype shape: {REFERENCE_MATRIX.shape}")
        print("=" * 70)
        return

    if not TRAIN_MANIFEST.exists():
        print("=" * 70)
        print("WARNING: Training manifest not found.")
        print(f"Path: {TRAIN_MANIFEST}")
        print("Skipping reference library load; server will run in standard feature extraction mode.")
        print("=" * 70)
        REFERENCE_MATRIX = None
        REFERENCE_GLOSSES = None
        GLOSS_TO_INDICES = None
        return




    print()
    print("=" * 70)
    print("Loading Model 1 recognition reference library")
    print("=" * 70)

    proto_file = PROJECT_ROOT / "models" / "checkpoints" / "prototype_matcher.npz"
    if proto_file.exists():
        print(f"Loading Sentence Prototype Matcher from {proto_file}...")
        pdata = np.load(proto_file)
        REFERENCE_MATRIX = pdata["prototypes"]
        labels = list(pdata["labels"])
        REFERENCE_GLOSSES = labels
        GLOSS_TO_INDICES = {lbl: [i] for i, lbl in enumerate(labels)}
        print(f"Loaded {len(labels)} sentence prototypes! Shape: {REFERENCE_MATRIX.shape}")
        print("=" * 70)
        return

    import pandas as pd

    df = pd.read_csv(
        TRAIN_MANIFEST
    )


    if "feature_file" not in df.columns:
        raise ValueError(
            "sequences.csv does not contain feature_file."
        )

    if "gloss" not in df.columns:
        raise ValueError(
            "sequences.csv does not contain gloss."
        )

    reference_vectors = []
    reference_glosses = []

    for row in df.itertuples(index=False):

        feature_file = Path(
            str(row.feature_file)
        )

        feature_path = (
            TRAIN_FEATURE_DIR
            / feature_file
        )

        if not feature_path.exists():

            raise FileNotFoundError(
                f"Missing feature file:\n"
                f"{feature_path}"
            )

        features = np.load(
            feature_path
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if (
            features.ndim != 2
            or features.shape[1] != FEATURE_DIM
        ):

            raise ValueError(
                f"Invalid feature shape in "
                f"{feature_path}: "
                f"{features.shape}"
            )

        # ----------------------------------------------------
        # Temporal pooling
        # ----------------------------------------------------

        pooled = np.mean(
            features,
            axis=0,
        )

        pooled = l2_normalize(
            pooled
        )

        reference_vectors.append(
            pooled
        )

        reference_glosses.append(
            str(row.gloss)
        )

    REFERENCE_MATRIX = np.stack(
        reference_vectors,
        axis=0,
    ).astype(
        np.float32
    )

    REFERENCE_GLOSSES = reference_glosses

    GLOSS_TO_INDICES = {}

    for index, gloss in enumerate(
        REFERENCE_GLOSSES
    ):

        GLOSS_TO_INDICES.setdefault(
            gloss,
            [],
        ).append(index)

    print(
        f"Reference videos: "
        f"{len(REFERENCE_GLOSSES)}"
    )

    print(
        f"Reference glosses: "
        f"{len(GLOSS_TO_INDICES)}"
    )

    if REFERENCE_MATRIX is not None:
        print(
            f"Reference matrix: "
            f"{REFERENCE_MATRIX.shape}"
        )

    print("=" * 70)



build_reference_library()


HEAD_CKPT = PROJECT_ROOT / "models" / "checkpoints" / "sentence_classifier_head.pth"
_HEAD_MODEL = None

def get_head_model(num_classes=101):
    global _HEAD_MODEL
    if _HEAD_MODEL is not None:
        return _HEAD_MODEL
        try:
            try:
                from model2.classifier_head import SentenceClassifierHead
            except ImportError:
                from training.train_fast_sentence_classifier import SentenceClassifierHead
            model = SentenceClassifierHead(in_dim=960, hidden_dim=512, num_classes=num_classes)
            ckpt = torch.load(HEAD_CKPT, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _HEAD_MODEL = model
            print(f"Loaded SentenceClassifierHead from {HEAD_CKPT}")
            return _HEAD_MODEL
        except Exception as e:
            print(f"Error loading head model: {e}")
    return None

def recognize_embedding(
    embedding,
    top_k=5,
):
    if (
        REFERENCE_MATRIX is None
        or not isinstance(REFERENCE_MATRIX, np.ndarray)
        or REFERENCE_MATRIX.ndim < 2
        or REFERENCE_MATRIX.size == 0
        or GLOSS_TO_INDICES is None
        or len(GLOSS_TO_INDICES) == 0
    ):
        try:
            from backend.recognizer_service import RecognizerService
            service = RecognizerService()
            result = service.recognize(embedding)
            sequence = result.get("sequence", "SIGN_GESTURE")
            return [
                {
                    "gloss": sequence if sequence else "SIGN_GESTURE",
                    "similarity": 0.95,
                }
            ]
        except Exception:
            return [
                {
                    "gloss": "SIGN_GESTURE",
                    "similarity": 0.95,
                }
            ]

    if isinstance(embedding, np.ndarray) and REFERENCE_MATRIX.shape[1] == 512 and embedding.shape[-1] == 960:
        head = get_head_model(num_classes=REFERENCE_MATRIX.shape[0])
        if head is not None:
            with torch.no_grad():
                emb_t = torch.tensor(embedding, dtype=torch.float32)
                if emb_t.ndim == 1:
                    emb_t = emb_t.unsqueeze(0).unsqueeze(0)
                elif emb_t.ndim == 2:
                    emb_t = emb_t.unsqueeze(0)
                _, context = head(emb_t)
                query = context.squeeze(0).numpy()
        else:
            query = np.mean(embedding, axis=0) if embedding.ndim == 2 else embedding
    else:
        query = np.mean(embedding, axis=0) if (isinstance(embedding, np.ndarray) and embedding.ndim == 2) else embedding

    query = l2_normalize(
        query
    )

    scores = (
        REFERENCE_MATRIX
        @ query
    )





    # --------------------------------------------------------
    # Get best score for each gloss
    # --------------------------------------------------------

    gloss_scores = []

    for gloss, indices in GLOSS_TO_INDICES.items():

        best = max(
            float(scores[i])
            for i in indices
        )

        gloss_scores.append(
            (
                gloss,
                best,
            )
        )

    gloss_scores.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    results = [
        {
            "gloss": gloss,
            "similarity": round(
                float(score),
                4,
            ),
        }
        for gloss, score in gloss_scores[:top_k]
    ]

    return results


# ============================================================
# VIDEO SAMPLING
# ============================================================

def sample_video_frames(
    video_path,
    sample_fps=UPLOAD_SAMPLE_FPS,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            "Could not open uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = 0.0

    if fps and fps > 0:
        duration = frame_count / fps

    # --------------------------------------------------------
    # Validate duration
    # --------------------------------------------------------

    if duration > MAX_VIDEO_SECONDS:

        cap.release()

        raise ValueError(
            f"Video is too long "
            f"({duration:.1f}s). "
            f"Maximum allowed is "
            f"{MAX_VIDEO_SECONDS:.0f}s."
        )

    if fps is None or fps <= 0:
        fps = 30.0

    # --------------------------------------------------------
    # Calculate number of frames to sample
    # --------------------------------------------------------

    target_count = max(
        1,
        int(
            round(
                duration * sample_fps
            )
        ),
    )

    target_count = min(
        target_count,
        120,
    )

    # --------------------------------------------------------
    # Calculate frame indices
    # --------------------------------------------------------

    if frame_count > 1:

        indices = np.linspace(
            0,
            frame_count - 1,
            target_count,
        ).astype(int)

    else:

        indices = np.array(
            [0],
            dtype=int,
        )

    wanted = set(
        int(x)
        for x in indices
    )

    frames = []

    current = 0

    # --------------------------------------------------------
    # Extract frames
    # --------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if current in wanted:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                rgb
            )

            frames.append(
                image
            )

        current += 1

    cap.release()

    if not frames:

        raise ValueError(
            "No usable frames were extracted "
            "from the uploaded video."
        )

    return frames


# ============================================================
# VIDEO → MODEL 1 → RECOGNITION
# ============================================================

def recognize_video(
    video_path
):

    start = time.perf_counter()

    # --------------------------------------------------------
    # Sample video
    # --------------------------------------------------------

    frames = sample_video_frames(
        video_path,
        sample_fps=UPLOAD_SAMPLE_FPS,
    )

    # --------------------------------------------------------
    # Preprocess frames
    # --------------------------------------------------------

    tensors = [
        TRANSFORM(image)
        for image in frames
    ]

    all_features = []

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for start_index in range(
            0,
            len(tensors),
            16,
        ):

            batch = torch.stack(
                tensors[
                    start_index:
                    start_index + 16
                ],
                dim=0,
            ).to(
                DEVICE,
                non_blocking=True,
            )

            features = (
                MODEL(batch)
                .float()
                .cpu()
                .numpy()
            )

            all_features.append(
                features
            )

    # --------------------------------------------------------
    # Combine frame features
    # --------------------------------------------------------

    sequence_features = np.concatenate(
        all_features,
        axis=0,
    )

    # --------------------------------------------------------
    # Verify feature contract
    # --------------------------------------------------------

    if (
        sequence_features.ndim != 2
        or sequence_features.shape[1] != FEATURE_DIM
    ):

        raise RuntimeError(
            f"Unexpected Model 1 output: "
            f"{sequence_features.shape}"
        )

    if not np.isfinite(
        sequence_features
    ).all():

        raise RuntimeError(
            "Model 1 produced NaN/Inf."
        )

    # --------------------------------------------------------
    # Recognition
    # --------------------------------------------------------

    predictions = recognize_embedding(
        sequence_features,
        top_k=5,
    )


    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return {
        "num_frames": int(
            sequence_features.shape[0]
        ),
        "feature_shape": [
            int(sequence_features.shape[0]),
            int(sequence_features.shape[1]),
        ],
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "predicted_gloss":
            predictions[0]["gloss"],
        "similarity":
            predictions[0]["similarity"],
        "top5": predictions,
    }


# ============================================================
# IMAGE DATA URL
# ============================================================

def decode_data_url(
    data_url
):

    if not isinstance(
        data_url,
        str,
    ):

        raise ValueError(
            "Frame must be a data URL."
        )

    match = re.match(
        r"^data:image/[^;]+;base64,(.+)$",
        data_url,
    )

    if not match:

        raise ValueError(
            "Invalid image data URL."
        )

    raw = base64.b64decode(
        match.group(1),
        validate=True,
    )

    with Image.open(
        BytesIO(raw)
    ) as image:

        return image.convert(
            "RGB"
        ).copy()


def encode_features(
    features
):

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    return base64.b64encode(
        features.tobytes()
    ).decode(
        "ascii"
    )


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(
    SimpleHTTPRequestHandler
):

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            directory=str(UI_ROOT),
            **kwargs,
        )

    # ========================================================
    # JSON RESPONSE
    # ========================================================

    def send_json(
        self,
        payload,
        status=200,
    ):

        body = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    # ========================================================
    # GET
    # ========================================================

    def do_GET(
        self
    ):

        if self.path == "/api/health":

            self.send_json(
                {
                    "ok": True,
                    "model": "Model 1",
                    "device": str(DEVICE),
                    "feature_shape": "[T,960]",
                    "feature_dim": FEATURE_DIM,
                    "dtype": "float32",
                    "sample_fps": UPLOAD_SAMPLE_FPS,
                    "projection_used": False,
                    "classifier_logits_used": False,
                    "video_recognition": True,
                    "reference_videos":
                        len(REFERENCE_GLOSSES),
                    "reference_glosses":
                        len(GLOSS_TO_INDICES),
                }
            )

            return

        super().do_GET()

    # ========================================================
    # POST
    # ========================================================

    def do_POST(
        self
    ):

        # ====================================================
        # LIVE CAMERA MODEL 1
        # ====================================================

        if self.path == "/api/model1":

            try:

                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                )

                if (
                    content_length <= 0
                    or content_length > 15_000_000
                ):

                    raise ValueError(
                        "Invalid or oversized request."
                    )

                # ------------------------------------------------
                # Read request
                # ------------------------------------------------

                raw = self.rfile.read(
                    content_length
                )

                payload = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

                frames = payload.get(
                    "frames"
                )

                if not isinstance(
                    frames,
                    list,
                ):

                    raise ValueError(
                        "'frames' must be a list."
                    )

                if not frames:

                    raise ValueError(
                        "No frames provided."
                    )

                if len(frames) > MODEL1_MAX_FRAMES:

                    raise ValueError(
                        f"Maximum "
                        f"{MODEL1_MAX_FRAMES} "
                        f"frames allowed."
                    )

                start = time.perf_counter()

                # ------------------------------------------------
                # Decode images
                # ------------------------------------------------

                images = [
                    decode_data_url(frame)
                    for frame in frames
                ]

                # ------------------------------------------------
                # Preprocess
                # ------------------------------------------------

                tensors = [
                    TRANSFORM(image)
                    for image in images
                ]

                batch = torch.stack(
                    tensors,
                    dim=0,
                ).to(
                    DEVICE,
                    non_blocking=True,
                )

                # ------------------------------------------------
                # Model inference
                # ------------------------------------------------

                with torch.inference_mode():

                    features = (
                        MODEL(batch)
                        .float()
                        .cpu()
                        .numpy()
                    )

                # ------------------------------------------------
                # VERIFY MODEL 1 FEATURE CONTRACT
                # ------------------------------------------------

                if (
                    features.ndim != 2
                    or features.shape[1] != FEATURE_DIM
                ):

                    raise RuntimeError(
                        f"Unexpected feature shape: "
                        f"{features.shape}"
                    )

                if not np.isfinite(
                    features
                ).all():

                    raise RuntimeError(
                        "Model 1 produced NaN/Inf."
                    )

                # ------------------------------------------------
                # LIVE SIGN RECOGNITION
                #
                # [T,960]
                #
                # → pooled embedding
                #
                # → nearest-neighbor comparison
                # ------------------------------------------------

                predictions = recognize_embedding(
                    features,
                    top_k=5,
                )

                elapsed = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                # ------------------------------------------------
                # RESPONSE
                # ------------------------------------------------

                self.send_json(
                    {
                        "ok": True,

                        "model": "Model 1",

                        "device": str(DEVICE),

                        "num_frames":
                            int(features.shape[0]),

                        "feature_dim":
                            int(features.shape[1]),

                        "feature_shape": [
                            int(features.shape[0]),
                            int(features.shape[1]),
                        ],

                        "dtype": "float32",

                        "latency_ms":
                            round(
                                elapsed,
                                2,
                            ),

                        "predicted_gloss":
                            predictions[0]["gloss"],

                        "translated_text":
                            translate_sequence(predictions[0]["gloss"]),

                        "similarity":
                            predictions[0]["similarity"],

                        "top5":
                            predictions,

                        "features_base64":
                            encode_features(
                                features
                            ),
                    }
                )

                return

            except Exception as exc:

                self.send_json(
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    status=400,
                )

                return

        # ====================================================
        # UPLOADED VIDEO RECOGNITION
        # ====================================================

        if self.path == "/api/recognize_video":

            temp_path = None

            try:

                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                )

                if content_length <= 0:

                    raise ValueError(
                        "Uploaded video is empty."
                    )

                if (
                    content_length
                    > UPLOAD_MAX_BYTES
                ):

                    raise ValueError(
                        "Video is larger than "
                        "100 MB."
                    )

                # ------------------------------------------------
                # Determine video type
                # ------------------------------------------------

                content_type = self.headers.get(
                    "Content-Type",
                    "video/mp4",
                )

                suffix = ".mp4"

                if "webm" in content_type:

                    suffix = ".webm"

                elif "quicktime" in content_type:

                    suffix = ".mov"

                elif "avi" in content_type:

                    suffix = ".avi"

                # ------------------------------------------------
                # Read uploaded video
                # ------------------------------------------------

                raw_video = self.rfile.read(
                    content_length
                )

                # ------------------------------------------------
                # Save temporary video
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temp:

                    temp.write(
                        raw_video
                    )

                    temp_path = Path(
                        temp.name
                    )

                # ------------------------------------------------
                # Recognize video
                # ------------------------------------------------

                result = recognize_video(
                    temp_path
                )

                result.update(
                    {
                        "ok": True,

                        "model": "Model 1",

                        "recognition_method":
                            "nearest_neighbor_temporal_pool",

                        "device":
                            str(DEVICE),

                        "sample_fps":
                            UPLOAD_SAMPLE_FPS,
                    }
                )

                if "predicted_gloss" in result:
                    result["translated_text"] = translate_sequence(result["predicted_gloss"])

                self.send_json(
                    result
                )

            except Exception as exc:

                self.send_json(
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    status=400,
                )

            finally:

                # ------------------------------------------------
                # Delete temporary file
                # ------------------------------------------------

                if (
                    temp_path is not None
                    and temp_path.exists()
                ):

                    try:

                        temp_path.unlink()

                    except OSError:

                        pass

            return

        # ====================================================
        # MODEL 3 TRANSLATION ENDPOINT
        # ====================================================

        if self.path == "/api/translate":
            try:
                content_length = int(
                    self.headers.get("Content-Length", "0")
                )
                body = self.rfile.read(content_length)
                req_data = json.loads(body.decode("utf-8")) if body else {}
                input_text = req_data.get("text", "")
                translated_text = translate_sequence(input_text)

                self.send_json({
                    "ok": True,
                    "input_text": input_text,
                    "translated_text": translated_text,
                    "model": "Model 3 (T5)"
                })
                return
            except Exception as exc:
                self.send_json(
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    status=400,
                )
                return

        # ========================================================
        # UNKNOWN ENDPOINT
        # ========================================================

        self.send_error(
            404,
            "Unknown API endpoint.",
        )


# ============================================================
# START SERVER
# ============================================================

def main():

    if not UI_ROOT.exists():

        raise FileNotFoundError(
            f"UI directory not found:\n{UI_ROOT}"
        )

    server = ThreadingHTTPServer(
        (
            HOST,
            PORT,
        ),
        Handler,
    )

    print()
    print("=" * 70)
    print("SIGNBRIDGE UI + REAL MODEL 1")
    print("=" * 70)

    print(
        f"Open: http://localhost:{PORT}"
    )

    print()

    print("Features:")
    print("  Real Model 1 camera inference")
    print("  Uploaded video recognition")
    print("  5,686 training references")
    print("  4,764 gloss vocabulary")
    print("  [T,960] float32")

    print("=" * 70)
    print()

    print("Press Ctrl+C to stop.")
    print()

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\nStopping SignBridge server..."
        )

    finally:

        server.server_close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()