import os
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================
# Paths
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR       = os.path.join(ROOT, "data", "continuous", "new_train")
VAL_DIR         = os.path.join(ROOT, "data", "continuous", "new_val")
CHECKPOINT_DIR  = os.path.join(ROOT, "models", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "temporal_classifier_best.pth")
CLASSES_OUT     = os.path.join(CHECKPOINT_DIR, "temporal_classifier_classes.json")

# ============================================================
# Hyperparameters
# ============================================================

INPUT_DIM    = 960
D_MODEL      = 256
NHEAD        = 8
NUM_LAYERS   = 4
DIM_FF       = 512
DROPOUT      = 0.2

EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Build class vocabulary from the dataset itself
# ============================================================

def build_vocab(train_dir, val_dir):
    all_classes = set()
    for directory in [train_dir, val_dir]:
        for fname in os.listdir(directory):
            if fname.endswith(".json"):
                fpath = os.path.join(directory, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                for lbl in labels:
                    all_classes.add(lbl)
    sorted_classes = sorted(all_classes)
    class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    return class_to_idx, idx_to_class

# ============================================================
# Dataset
# ============================================================

class TemporalClassifierDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, augment=False, preload=True):
        self.data_dir     = data_dir
        self.class_to_idx = class_to_idx
        self.augment      = augment
        self.samples = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".npy")
        )
        if not self.samples:
            raise ValueError(f"No .npy files found in {data_dir}")

        self.preloaded = []
        if preload:
            print(f"Preloading {len(self.samples)} samples from {os.path.basename(data_dir)} into RAM...")
            for npy_file in self.samples:
                json_file = npy_file.replace(".npy", ".json")
                feat_path  = os.path.join(self.data_dir, npy_file)
                label_path = os.path.join(self.data_dir, json_file)

                features = np.load(feat_path).astype(np.float32)
                with open(label_path, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                label_idx = self.class_to_idx[labels[0]]
                self.preloaded.append((features, label_idx))
            print(f"Preloaded {len(self.samples)} samples successfully!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preloaded:
            features, label_idx = self.preloaded[idx]
            features = features.copy()
        else:
            npy_file  = self.samples[idx]
            json_file = npy_file.replace(".npy", ".json")
            feat_path  = os.path.join(self.data_dir, npy_file)
            label_path = os.path.join(self.data_dir, json_file)

            features = np.load(feat_path).astype(np.float32)
            with open(label_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            label_idx = self.class_to_idx[labels[0]]

        if self.augment:
            # Gaussian noise
            features = features + np.random.randn(*features.shape).astype(np.float32) * 0.01
            # Random time dropout
            T = features.shape[0]
            if T > 2:
                keep_mask = np.random.rand(T) > 0.1
                if keep_mask.sum() > 0:
                    features = features[keep_mask]
            # Speed perturbation
            T = features.shape[0]
            scale = np.random.uniform(0.85, 1.15)
            new_T = max(2, int(T * scale))
            if new_T != T:
                indices = np.linspace(0, T - 1, new_T).astype(int)
                features = features[indices]

        return torch.tensor(features, dtype=torch.float32), label_idx

# ============================================================
# Collate
# ============================================================

def collate_fn(batch):
    features_list, labels_list = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    padded = torch.zeros(len(features_list), lengths.max().item(), features_list[0].size(-1))
    for i, feat in enumerate(features_list):
        T = feat.size(0)
        padded[i, :T] = feat
    padding_mask = torch.zeros(len(features_list), lengths.max().item(), dtype=torch.bool)
    for i, T in enumerate(lengths.tolist()):
        padding_mask[i, T:] = True
    labels = torch.tensor(labels_list, dtype=torch.long)
    return padded, padding_mask, labels, lengths

# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

# ============================================================
# Model 2 — Temporal Transformer Classifier
# ============================================================

class TemporalTransformerClassifier(nn.Module):
    """
    Person 2's Model 2.
    Input  : [B, T, 960] - feature sequences from Person 1's MobileNetV3
    Output : [B, num_classes] logits
    Contract output (inference): {"sequence": "SIGN_NAME"}
    """
    def __init__(
        self,
        input_dim=960,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=4764,
        dropout=0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)

# ============================================================
# Label Smoothing Loss
# ============================================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        if targets.dim() == 1:
            with torch.no_grad():
                smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
                smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            return -(smooth_labels * log_probs).sum(dim=-1).mean()
        else:
            # Soft targets from MixUp
            return -(targets * log_probs).sum(dim=-1).mean()

# ============================================================
# Training/Evaluation functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, alpha=0.2):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, padding_mask, labels, _ in loader:
        features     = features.to(device)
        padding_mask = padding_mask.to(device)
        labels       = labels.to(device)

        # Apply MixUp augmentation if alpha > 0
        if alpha > 0 and random.random() < 0.5:
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(features.size(0)).to(device)
            mixed_features = lam * features + (1 - lam) * features[index]
            mixed_mask = padding_mask & padding_mask[index]

            target_a = F.one_hot(labels, num_classes=criterion.num_classes).float()
            target_b = F.one_hot(labels[index], num_classes=criterion.num_classes).float()
            mixed_targets = lam * target_a + (1 - lam) * target_b

            optimizer.zero_grad()
            logits = model(mixed_features, mixed_mask)
            loss = criterion(logits, mixed_targets)
        else:
            optimizer.zero_grad()
            logits = model(features, padding_mask)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    top5 = 0
    total = 0
    for features, padding_mask, labels, _ in loader:
        features     = features.to(device)
        padding_mask = padding_mask.to(device)
        labels       = labels.to(device)
        logits = model(features, padding_mask)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        top5_pred   = logits.topk(5, dim=-1).indices
        top5       += (top5_pred == labels.unsqueeze(1)).any(dim=-1).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total, top5 / total

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Person 2 - Temporal Transformer Classifier (Model 2)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print("\nBuilding class vocabulary...")
    class_to_idx, idx_to_class = build_vocab(TRAIN_DIR, VAL_DIR)
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved class mapping to {CLASSES_OUT}")

    train_ds = TemporalClassifierDataset(TRAIN_DIR, class_to_idx, augment=True)
    val_ds   = TemporalClassifierDataset(VAL_DIR,   class_to_idx, augment=False)
    print(f"\nTrain: {len(train_ds)} sequences  |  Val: {len(val_ds)} sequences")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = TemporalTransformerClassifier(
        input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        num_classes=num_classes, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = LabelSmoothingLoss(num_classes, smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    print()
    print(f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VlLoss':>8}  {'VlAcc':>7}  {'Top5':>7}  {'LR':>9}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, alpha=MIXUP_ALPHA)
        vl_loss, vl_acc, vl_top5 = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        marker = " *" if vl_acc > best_val_acc else ""
        print(
            f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%  "
            f"{vl_loss:>8.4f}  {vl_acc*100:>6.2f}%  {vl_top5*100:>6.2f}%  "
            f"{lr_now:>9.2e}  [{elapsed:.0f}s]{marker}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": vl_acc,
                "val_top5": vl_top5,
                "num_classes": num_classes,
                "config": {
                    "input_dim": INPUT_DIM,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FF,
                    "dropout": DROPOUT,
                    "num_classes": num_classes,
                },
            }, CHECKPOINT_PATH)

    print()
    print("=" * 60)
    print(f"Training complete. Best val accuracy: {best_val_acc * 100:.2f}%")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
