import torch

from models.cnn_baseline import CNNBaseline


def main() -> None:
    num_classes = 39

    model = CNNBaseline(num_classes=num_classes)

    # Test a normal image batch.
    x = torch.randn(
        4,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    with torch.no_grad():
        y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    expected_shape = (4, num_classes)

    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}, "
            f"got {tuple(y.shape)}"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Number of parameters:", parameter_count)
    print("CNN baseline test successful.")


if __name__ == "__main__":
    main()