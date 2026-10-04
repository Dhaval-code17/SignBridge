import torch

from models.mobilenet_model import MobileNetV3SignRecognizer


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MobileNetV3SignRecognizer(
        num_classes=39,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    model.eval()

    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        logits, features = model(x)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Input:", tuple(x.shape))
    print("Logits:", tuple(logits.shape))
    print("Features:", tuple(features.shape))
    print("Feature dimension:", model.feature_dim)

    assert tuple(logits.shape) == (2, 39)
    assert tuple(features.shape) == (2, 960)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)

    print("MobileNetV3 model test successful.")


if __name__ == "__main__":
    main()