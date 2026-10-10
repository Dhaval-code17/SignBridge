"""
Model 1: Fine-tuning MobileNetV3 visual feature extractor on 101 ISL sentence classes.
"""

import json, random, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms

ROOT        = Path(__file__).resolve().parents[1]
VIDEO_DIR   = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
CACHE_DIR   = ROOT / "data" / "sentence_frames_cache"
CKPT_DIR    = ROOT / "models" / "checkpoints"
MODEL1_CKPT = CKPT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"
SENT_CKPT   = CKPT_DIR / "mobilenet_v3_sentence_finetuned.pth"
PROTO_OUT   = CKPT_DIR / "prototype_matcher.npz"
CLASSES_OUT = CKPT_DIR / "temporal_classifier_classes.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE       = 224
FRAMES_PER_VIDEO = 16
FRAMES_PER_CLIP  = 8
EPOCHS           = 30
BATCH_SIZE       = 8
LR_HEAD          = 1e-3
LR_BACKBONE      = 1e-4
WEIGHT_DECAY     = 1e-4
GRAD_CLIP        = 5.0
VAL_SPLIT        = 0.15
SEED             = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TRANSFORM_TRAIN = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
TRANSFORM_VAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


class SentenceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        bb = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = bb.features
        self.avgpool  = bb.avgpool
        self.head = nn.Sequential(
            nn.Linear(960, 512), nn.Hardswish(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.head(feat), feat


def load_model(num_classes):
    model = SentenceClassifier(num_classes)
    if MODEL1_CKPT.exists():
        ckpt  = torch.load(MODEL1_CKPT, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        ms    = model.state_dict()
        matched = 0
        for ck, cv in state.items():
            mk = ck.replace("backbone.", "features.")
            if mk in ms and ms[mk].shape == cv.shape:
                ms[mk] = cv
                matched += 1
        model.load_state_dict(ms, strict=False)
        print(f"[Model 1] Loaded {matched} matched ISL backbone params from checkpoint")
    else:
        print("[Model 1] Using ImageNet pretrained weights")

    for i, layer in enumerate(model.features):
        for p in layer.parameters():
            p.requires_grad = (i >= 12)
    for p in model.head.parameters():
        p.requires_grad = True

    return model.to(DEVICE)


def main():
    print("SignBridge Model 1 Fine-Tuning")
    if not VIDEO_DIR.exists():
        print(f"Video directory {VIDEO_DIR} not found. Skipping training run.")
        return

    exts = {".mp4",".MP4",".mov",".MOV",".avi",".AVI"}
    all_items = []
    for sd in sorted(VIDEO_DIR.iterdir()):
        if not sd.is_dir(): continue
        label = sd.name.strip().lower()
        for vf in sd.iterdir():
            if vf.suffix in exts:
                all_items.append((vf, label))

    classes = sorted({lbl for _, lbl in all_items})
    c2i = {l:i for i,l in enumerate(classes)}
    NC  = len(classes)
    print(f"Loaded {NC} classes for fine-tuning.")

if __name__ == "__main__":
    main()
