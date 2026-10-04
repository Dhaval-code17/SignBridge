import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


def main() -> None:
    weights = MobileNet_V3_Large_Weights.DEFAULT

    model = mobilenet_v3_large(
        weights=weights,
    )

    print("Model created successfully.")
    print("Classifier:")
    print(model.classifier)

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    assert tuple(y.shape) == (2, 1000)

    print("Pretrained MobileNetV3 test successful.")


if __name__ == "__main__":
    main()