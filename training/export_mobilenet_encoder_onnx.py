from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from models.mobilenet_model import MobileNetV3SignRecognizer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "checkpoints"
    / "mobilenet_v3_stage2_best.pth"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "onnx"
)

ONNX_PATH = (
    OUTPUT_DIR
    / "mobilenet_v3_stage2_encoder.onnx"
)

INPUT_NAME = "input"
OUTPUT_NAME = "features"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# ENCODER WRAPPER
# ============================================================

class MobileNetEncoder(nn.Module):
    """
    Exposes only the 960-D visual feature extractor.
    """

    def __init__(self, model):
        super().__init__()

        self.backbone = model.backbone
        self.pool = model.pool

    def forward(self, x):

        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("MOBILENETV3 STAGE-2 ONNX EXPORT")
print("=" * 65)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

num_classes = len(
    checkpoint["class_to_idx"]
)

feature_dim = checkpoint[
    "feature_dim"
]

print(
    f"Checkpoint epoch: {checkpoint['epoch']}"
)

print(
    f"Checkpoint validation Macro-F1: "
    f"{checkpoint['val_f1']:.4f}"
)

print(
    f"Feature dimension: {feature_dim}"
)

if feature_dim != 960:
    raise RuntimeError(
        f"Expected feature dimension 960, "
        f"got {feature_dim}"
    )


model = MobileNetV3SignRecognizer(
    num_classes=num_classes,
    pretrained=False,
    freeze_backbone=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


encoder = MobileNetEncoder(
    model
)

encoder.eval()


# ============================================================
# DUMMY INPUT
# ============================================================

dummy_input = torch.randn(
    2,
    3,
    224,
    224,
    dtype=torch.float32,
    device=device,
)

with torch.no_grad():
    pytorch_output = encoder(
        dummy_input
    )

print(
    f"\nPyTorch input shape: "
    f"{tuple(dummy_input.shape)}"
)

print(
    f"PyTorch output shape: "
    f"{tuple(pytorch_output.shape)}"
)

if tuple(pytorch_output.shape) != (
    2,
    960,
):
    raise RuntimeError(
        "Unexpected PyTorch encoder output shape."
    )


# ============================================================
# EXPORT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "\nExporting ONNX..."
)

torch.onnx.export(
    encoder,
    dummy_input,
    str(ONNX_PATH),
    input_names=[INPUT_NAME],
    output_names=[OUTPUT_NAME],
    opset_version=17,
    dynamic_axes={
        INPUT_NAME: {
            0: "batch"
        },
        OUTPUT_NAME: {
            0: "batch"
        },
    },
    do_constant_folding=True,
    dynamo=False,
)

print(
    f"ONNX saved:\n{ONNX_PATH}"
)


# ============================================================
# CHECK ONNX MODEL
# ============================================================

print(
    "\nChecking ONNX model..."
)

onnx_model = onnx.load(
    str(ONNX_PATH)
)

onnx.checker.check_model(
    onnx_model
)

print(
    "ONNX model structure: valid"
)


# ============================================================
# ONNX RUNTIME
# ============================================================

print(
    "\nRunning ONNXRuntime verification..."
)

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print(
    f"ONNX input name: "
    f"{input_info.name}"
)

print(
    f"ONNX output name: "
    f"{output_info.name}"
)

onnx_output = session.run(
    [OUTPUT_NAME],
    {
        INPUT_NAME:
        dummy_input.numpy()
    },
)[0]


# ============================================================
# NUMERICAL COMPARISON
# ============================================================

pytorch_np = (
    pytorch_output
    .detach()
    .cpu()
    .numpy()
)

onnx_np = np.asarray(
    onnx_output
)

print(
    f"\nONNX output shape: "
    f"{onnx_np.shape}"
)

if onnx_np.shape != (
    2,
    960,
):
    raise RuntimeError(
        "Unexpected ONNX output shape."
    )

max_abs_diff = np.max(
    np.abs(
        pytorch_np - onnx_np
    )
)

mean_abs_diff = np.mean(
    np.abs(
        pytorch_np - onnx_np
    )
)

pytorch_flat = (
    pytorch_np
    .reshape(-1)
)

onnx_flat = (
    onnx_np
    .reshape(-1)
)

denom = (
    np.linalg.norm(pytorch_flat)
    * np.linalg.norm(onnx_flat)
)

if denom == 0:
    cosine_similarity = 0.0
else:
    cosine_similarity = float(
        np.dot(
            pytorch_flat,
            onnx_flat,
        )
        / denom
    )


print(
    f"Max absolute difference : "
    f"{max_abs_diff:.8f}"
)

print(
    f"Mean absolute difference: "
    f"{mean_abs_diff:.8f}"
)

print(
    f"Cosine similarity       : "
    f"{cosine_similarity:.8f}"
)


# ============================================================
# PASS/FAIL
# ============================================================

# Numerical tolerance for this export.
PASS = (
    max_abs_diff < 1e-4
    and mean_abs_diff < 1e-5
    and cosine_similarity > 0.9999
)

if not PASS:
    raise RuntimeError(
        "PyTorch vs ONNX numerical verification failed."
    )


# ============================================================
# DYNAMIC BATCH TEST
# ============================================================

print(
    "\nTesting dynamic batch dimension..."
)

dynamic_input = np.random.randn(
    5,
    3,
    224,
    224,
).astype(
    np.float32
)

dynamic_output = session.run(
    [OUTPUT_NAME],
    {
        INPUT_NAME:
        dynamic_input
    },
)[0]

print(
    f"Dynamic input shape : "
    f"{dynamic_input.shape}"
)

print(
    f"Dynamic output shape: "
    f"{dynamic_output.shape}"
)

if dynamic_output.shape != (
    5,
    960,
):
    raise RuntimeError(
        "Dynamic batch test failed."
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 65
)

print(
    "ONNX EXPORT AND VERIFICATION PASSED"
)

print(
    "=" * 65
)

print(
    f"ONNX file:\n{ONNX_PATH}"
)

print(
    "Input : [B, 3, 224, 224]"
)

print(
    "Output: [B, 960]"
)

print(
    "Dynamic batch: verified"
)

print(
    "PyTorch/ONNX agreement: verified"
)