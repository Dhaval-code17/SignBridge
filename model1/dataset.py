"""
Model 1: Frame Dataset & Video Sampling Utilities for SignBridge.
Handles video frame decoding, duration-aware sampling, and frame caching.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224

TRANSFORM_VAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def sample_video_frames(video_path, num_frames=16):
    """
    Uniformly samples num_frames from a video file.
    Returns:
        List of [H, W, C] BGR frame arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        total_frames = len(frames)
        if total_frames == 0:
            return []
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    return frames


class CachedSentenceFramesDataset(Dataset):
    """Dataset for pre-cached video frames."""
    def __init__(self, samples, cache_dir, transform=None, frames_per_clip=8):
        self.samples = samples
        self.cache_dir = cache_dir
        self.transform = transform or TRANSFORM_VAL
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_name, label = self.samples[idx]
        npy_path = self.cache_dir / f"{video_name}.npy"
        cached_frames = np.load(npy_path)

        total_cached = len(cached_frames)
        if total_cached >= self.frames_per_clip:
            indices = sorted(np.random.choice(total_cached, self.frames_per_clip, replace=False))
        else:
            indices = np.linspace(0, total_cached - 1, self.frames_per_clip, dtype=int)

        selected = cached_frames[indices]
        tensors = [self.transform(frame) for frame in selected]
        clip_tensor = torch.stack(tensors, dim=0)

        return clip_tensor, label
