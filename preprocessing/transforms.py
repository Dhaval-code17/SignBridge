import cv2
import numpy as np
import albumentations as A
import torch
from torchvision.models import MobileNet_V3_Large_Weights


IMAGE_SIZE = 224

# Official ImageNet normalization used by the
# pretrained MobileNetV3 weights.
WEIGHTS = MobileNet_V3_Large_Weights.DEFAULT
MEAN = np.array(WEIGHTS.transforms().mean, dtype=np.float32)
STD = np.array(WEIGHTS.transforms().std, dtype=np.float32)


def build_train_transform():
    """
    Augmentation used only during training.
    """
    return A.Compose(
        [
            A.Resize(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ),

            A.HorizontalFlip(
                p=0.0
            ),

            A.Rotate(
                limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.1,
            ),
        ]
    )


def build_eval_transform():
    """
    Deterministic preprocessing for validation/test.
    """
    return A.Compose(
        [
            A.Resize(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ),
        ]
    )


def preprocess_frame(
    frame_rgb: np.ndarray,
    transform: A.Compose,
) -> torch.Tensor:
    """
    Convert an RGB uint8 frame into a normalized CHW tensor.

    Input:
        [H, W, 3]

    Output:
        [3, 224, 224]
    """
    if frame_rgb.dtype != np.uint8:
        raise TypeError("Expected uint8 RGB frame.")

    result = transform(image=frame_rgb)
    image = result["image"].astype(np.float32) / 255.0

    image = (image - MEAN) / STD

    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image).float()