from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "processed" / "final_dev_train.csv"
VAL_CSV = ROOT / "data" / "processed" / "final_dev_val.csv"
CLASSES_JSON = ROOT / "configs" / "classes_final.json"

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"
)

HISTORY_PATH = (
    ROOT
    / "models"
    / "evaluation"
    / "full_vocab_contrastive_history.csv"
)


# ============================================================
# TRAINING CONFIG
# ============================================================

SEED = 42

NUM_FRAMES = 16
IMAGE_SIZE = 224

BATCH_SIZE = 2
NUM_WORKERS = 0

MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2

FRAME_CHUNK = 8

FEATURE_DIM = 960
PROJECTION_DIM = 128

BACKBONE_LR = 1e-5
PROJECTION_LR = 1e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07
GRAD_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("SIGNBRIDGE — FULL VOCABULARY CONTRASTIVE TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"PyTorch: {torch.__version__}"
)


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    CLASSES_JSON,
    "r",
    encoding="utf-8",
) as f:
    class_data = json.load(f)

class_to_idx = class_data["class_to_idx"]

NUM_CLASSES = len(class_to_idx)

assert NUM_CLASSES == 4764, (
    f"Expected 4764 classes, got {NUM_CLASSES}"
)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

required_columns = {
    "uid",
    "gloss",
    "video_path",
}

for name, dataframe in [
    ("train", train_df),
    ("validation", val_df),
]:

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"{name} CSV missing columns: {missing}"
        )


train_df["gloss"] = (
    train_df["gloss"]
    .astype(str)
    .str.strip()
)

val_df["gloss"] = (
    val_df["gloss"]
    .astype(str)
    .str.strip()
)


# ============================================================
# LABEL VALIDATION
# ============================================================

for name, dataframe in [
    ("train", train_df),
    ("validation", val_df),
]:

    unknown = (
        set(dataframe["gloss"])
        - set(class_to_idx)
    )

    if unknown:
        raise ValueError(
            f"{name} contains unknown classes: "
            f"{sorted(unknown)[:20]}"
        )


# ============================================================
# VIDEO PATH
# ============================================================

def resolve_video_path(value: str) -> Path:

    path = Path(str(value))

    if path.is_absolute():
        return path

    candidates = [
        ROOT / path,
        ROOT / "data" / "raw" / "CISLR" / path,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return path


# ============================================================
# VERIFY FILES
# ============================================================

print()
print("Checking training videos...")

missing_train = []

for value in train_df["video_path"]:

    path = resolve_video_path(value)

    if not path.exists():
        missing_train.append(str(path))


print(
    f"Training files checked: {len(train_df)}"
)

print(
    f"Missing training files: {len(missing_train)}"
)

if missing_train:
    for item in missing_train[:20]:
        print(item)

    raise FileNotFoundError(
        "Training video files are missing."
    )


print()
print("Checking validation videos...")

missing_val = []

for value in val_df["video_path"]:

    path = resolve_video_path(value)

    if not path.exists():
        missing_val.append(str(path))


print(
    f"Validation files checked: {len(val_df)}"
)

print(
    f"Missing validation files: {len(missing_val)}"
)

if missing_val:

    for item in missing_val[:20]:
        print(item)

    raise FileNotFoundError(
        "Validation video files are missing."
    )


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3
            )
        ],
        p=0.10,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


val_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


# ============================================================
# FRAME SAMPLING
# ============================================================

def sample_frames(
    video_path: Path,
    num_frames: int,
) -> list[Image.Image]:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frames = []

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No readable frames: {video_path}"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        num=num_frames,
        dtype=np.int64,
    )

    return [
        Image.fromarray(
            frames[int(index)]
        )
        for index in indices
    ]


# ============================================================
# DATASET
# ============================================================

class ContrastiveSignDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: dict[str, int],
        training: bool,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.class_to_idx = class_to_idx

        self.training = training

        self.transform = (
            train_transform
            if training
            else val_transform
        )

    def __len__(self):
        return len(self.df)

    def _make_view(
        self,
        frames,
    ):

        tensors = []

        for frame in frames:

            tensors.append(
                self.transform(frame)
            )

        return torch.stack(
            tensors,
            dim=0,
        )

    def __getitem__(self, index):

        row = self.df.iloc[index]

        video_path = resolve_video_path(
            row["video_path"]
        )

        frames = sample_frames(
            video_path,
            NUM_FRAMES,
        )

        if self.training:

            view_1 = self._make_view(
                frames
            )

            view_2 = self._make_view(
                frames
            )

        else:

            view_1 = self._make_view(
                frames
            )

            view_2 = self._make_view(
                frames
            )

        label = self.class_to_idx[
            row["gloss"]
        ]

        return (
            view_1,
            view_2,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================
# LOADERS
# ============================================================

train_dataset = ContrastiveSignDataset(
    train_df,
    class_to_idx,
    training=True,
)

val_dataset = ContrastiveSignDataset(
    val_df,
    class_to_idx,
    training=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# MODEL
# ============================================================

class SignBridgeContrastiveModel(nn.Module):

    def __init__(self):

        super().__init__()

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
        )

        base = models.mobilenet_v3_large(
            weights=weights
        )

        self.backbone = base.features

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.feature_dim = FEATURE_DIM

        self.projection = nn.Sequential(
            nn.Linear(
                FEATURE_DIM,
                256,
            ),
            nn.Hardswish(),

            nn.Linear(
                256,
                PROJECTION_DIM,
            ),
        )

    def encode_frames(
        self,
        frames: torch.Tensor,
    ):

        outputs = []

        for start in range(
            0,
            frames.shape[0],
            FRAME_CHUNK,
        ):

            chunk = frames[
                start:start + FRAME_CHUNK
            ]

            x = self.backbone(
                chunk
            )

            x = self.avgpool(x)

            x = torch.flatten(
                x,
                1,
            )

            outputs.append(x)

        return torch.cat(
            outputs,
            dim=0,
        )

    def encode_video(
        self,
        video: torch.Tensor,
    ):

        batch_size, num_frames, c, h, w = (
            video.shape
        )

        frames = video.reshape(
            batch_size * num_frames,
            c,
            h,
            w,
        )

        frame_features = self.encode_frames(
            frames
        )

        frame_features = frame_features.reshape(
            batch_size,
            num_frames,
            self.feature_dim,
        )

        # IMPORTANT:
        # this is the feature representation
        # consumed by Person 2.
        video_features = (
            frame_features.mean(dim=1)
        )

        return (
            video_features,
            frame_features,
        )

    def forward(
        self,
        video,
    ):

        video_features, frame_features = (
            self.encode_video(video)
        )

        projected = self.projection(
            video_features
        )

        return (
            projected,
            video_features,
            frame_features,
        )


model = SignBridgeContrastiveModel().to(
    DEVICE
)


# ============================================================
# FREEZE EARLY MOBILE NET BLOCKS
# ============================================================

for parameter in model.backbone.parameters():
    parameter.requires_grad = False


# Train the final six MobileNet blocks.
for block in list(
    model.backbone.children()
)[-6:]:

    for parameter in block.parameters():
        parameter.requires_grad = True


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

class SupervisedContrastiveLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
    ):

        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ):

        # Normalize embeddings.
        # Contrastive logits/loss are computed in FP32 to avoid
        # FP16 overflow during masked_fill/logsumexp.
        features = F.normalize(
            features.float(),
            dim=1,
        )

        similarity = torch.matmul(
            features,
            features.T,
        )

        # Force contrastive logits to FP32 after CUDA autocast.
        similarity = similarity.float()

        similarity = (
            similarity
            / self.temperature
        )

        n = features.size(0)

        device = features.device

        logits_mask = torch.ones(
            (n, n),
            dtype=torch.bool,
            device=device,
        )

        logits_mask.fill_diagonal_(False)

        # Same gloss = positive.
        positive_mask = (
            labels.unsqueeze(0)
            == labels.unsqueeze(1)
        )

        # Remove self-comparisons.
        positive_mask = (
            positive_mask
            & logits_mask
        )

        # Every video has two augmented views.
        # Therefore even singleton classes have
        # a guaranteed positive pair.
        instance_ids = torch.arange(
            n // 2,
            device=device,
        ).repeat(2)

        same_instance = (
            instance_ids.unsqueeze(0)
            == instance_ids.unsqueeze(1)
        )

        positive_mask = (
            positive_mask
            | (
                same_instance
                & logits_mask
            )
        )

        # Numerical stability.
        logits = similarity.masked_fill(
            ~logits_mask,
            -1e9,
        )

        log_prob = (
            logits
            - torch.logsumexp(
                logits,
                dim=1,
                keepdim=True,
            )
        )

        positive_count = (
            positive_mask.sum(
                dim=1
            ).clamp_min(1)
        )

        mean_log_prob_pos = (
            (
                positive_mask.float()
                * log_prob
            ).sum(dim=1)
            / positive_count
        )

        loss = (
            -mean_log_prob_pos
        ).mean()

        return loss


criterion = SupervisedContrastiveLoss(
    temperature=TEMPERATURE
)


# ============================================================
# OPTIMIZER
# ============================================================

backbone_parameters = [
    parameter
    for parameter in model.backbone.parameters()
    if parameter.requires_grad
]

projection_parameters = list(
    model.projection.parameters()
)

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR,
        },
        {
            "params": projection_parameters,
            "lr": PROJECTION_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
)


# ============================================================
# AMP
# ============================================================

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    # Keep frozen blocks in eval mode.
    for block in list(
        model.backbone.children()
    )[:-6]:

        block.eval()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False,
    )

    for view1, view2, labels in progress:

        view1 = view1.to(
            DEVICE,
            non_blocking=True,
        )

        view2 = view2.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                z1, _, _ = model(
                    view1
                )

                z2, _, _ = model(
                    view2
                )

                features = torch.cat(
                    [z1, z2],
                    dim=0,
                )

                contrastive_labels = (
                    torch.cat(
                        [labels, labels],
                        dim=0,
                    )
                )

                loss = criterion(
                    features,
                    contrastive_labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            z1, _, _ = model(
                view1
            )

            z2, _, _ = model(
                view2
            )

            features = torch.cat(
                [z1, z2],
                dim=0,
            )

            contrastive_labels = (
                torch.cat(
                    [labels, labels],
                    dim=0,
                )
            )

            loss = criterion(
                features,
                contrastive_labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        running_loss
        / len(train_dataset)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    )

    for view1, view2, labels in progress:

        view1 = view1.to(
            DEVICE,
            non_blocking=True,
        )

        view2 = view2.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                z1, _, _ = model(
                    view1
                )

                z2, _, _ = model(
                    view2
                )

                features = torch.cat(
                    [z1, z2],
                    dim=0,
                )

                contrastive_labels = (
                    torch.cat(
                        [labels, labels],
                        dim=0,
                    )
                )

                loss = criterion(
                    features,
                    contrastive_labels,
                )

        else:

            z1, _, _ = model(
                view1
            )

            z2, _, _ = model(
                view2
            )

            features = torch.cat(
                [z1, z2],
                dim=0,
            )

            contrastive_labels = (
                torch.cat(
                    [labels, labels],
                    dim=0,
                )
            )

            loss = criterion(
                features,
                contrastive_labels,
            )

        running_loss += (
            loss.item()
            * labels.size(0)
        )

    return (
        running_loss
        / len(val_dataset)
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    epoch,
    train_loss,
    val_loss,
):

    torch.save(
        {
            "stage":
                "full_vocabulary_contrastive",

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "backbone_state_dict":
                model.backbone.state_dict(),

            "projection_state_dict":
                model.projection.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "class_to_idx":
                class_to_idx,

            "num_classes":
                NUM_CLASSES,

            "num_frames":
                NUM_FRAMES,

            "feature_dim":
                FEATURE_DIM,

            "projection_dim":
                PROJECTION_DIM,

            "frame_chunk":
                FRAME_CHUNK,

            "temperature":
                TEMPERATURE,

            "image_size":
                IMAGE_SIZE,

            "seed":
                SEED,
        },
        BEST_CHECKPOINT,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    f"Training videos: {len(train_dataset)}"
)

print(
    f"Validation videos: {len(val_dataset)}"
)

print(
    f"Vocabulary: {NUM_CLASSES} classes"
)

print(
    f"Frames/video: {NUM_FRAMES}"
)

print(
    f"Visual feature dimension: {FEATURE_DIM}"
)

print(
    f"Projection dimension: {PROJECTION_DIM}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Frame chunk: {FRAME_CHUNK}"
)


print()
print("=" * 70)
print("STARTING CONTRASTIVE TRAINING")
print("=" * 70)


best_val_loss = float("inf")
epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    print()
    print(
        f"================ EPOCH "
        f"{epoch}/{MAX_EPOCHS} ================"
    )

    train_loss = train_one_epoch()

    val_loss = validate()

    scheduler.step(
        val_loss
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "backbone_lr":
                optimizer.param_groups[0]["lr"],
            "projection_lr":
                optimizer.param_groups[1]["lr"],
        }
    )

    print()
    print(
        f"Train contrastive loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Val contrastive loss:   "
        f"{val_loss:.4f}"
    )

    print(
        f"Backbone LR: "
        f"{optimizer.param_groups[0]['lr']:.2e}"
    )

    print(
        f"Projection LR: "
        f"{optimizer.param_groups[1]['lr']:.2e}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        save_checkpoint(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

        print()
        print(
            "NEW BEST CONTRASTIVE CHECKPOINT SAVED"
        )

        print(
            BEST_CHECKPOINT
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            break


# ============================================================
# HISTORY
# ============================================================

HISTORY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame(history).to_csv(
    HISTORY_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONTRASTIVE TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best checkpoint:\n"
    f"{BEST_CHECKPOINT}"
)

print(
    f"Training history:\n"
    f"{HISTORY_PATH}"
)

print()
print(
    "MODEL 1 FEATURE CONTRACT:"
)

print(
    "[T, 960] float32 ordered visual features"
)

print(
    "These features remain compatible with Person 2."
)

print("=" * 70)