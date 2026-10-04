import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from training.train_temporal_classifier import TemporalTransformerClassifier, INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT

def export_onnx():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(root, "models", "checkpoints", "temporal_classifier_best.pth")
    output_dir = os.path.join(root, "models", "model2")
    output_path = os.path.join(output_dir, "model2.onnx")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    num_classes = checkpoint.get("num_classes", 4764)
    config = checkpoint.get("config", {
        "input_dim": INPUT_DIM,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FF,
        "dropout": DROPOUT,
        "num_classes": num_classes,
    })

    model = TemporalTransformerClassifier(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Dummy input sequence [batch_size=1, T=30, feature_dim=960]
    dummy_input = torch.randn(1, 30, config["input_dim"], dtype=torch.float32)

    print(f"Exporting ONNX model to: {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"ONNX export successful! Model saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    export_onnx()
