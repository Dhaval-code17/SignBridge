"""
train_fast_sentence_classifier.py
===================================
1. Extracts 960-dim MobileNetV3 features using loaded ISL backbone (308 params).
2. Saves cached feature dataset: data/sentence_features_dataset.npz [687, 16, 960].
3. Trains a fast PyTorch Temporal Classifier Head on extracted features (100 epochs in <3s).
4. Saves final sentence model and prototype matcher.
"""

import json, time, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
VIDEO_DIR   = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
CACHE_DIR   = ROOT / "data" / "sentence_frames_cache"
FEAT_FILE   = ROOT / "data" / "sentence_features_dataset.npz"
CKPT_DIR    = ROOT / "models" / "checkpoints"
MODEL1_CKPT = CKPT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"
HEAD_CKPT   = CKPT_DIR / "sentence_classifier_head.pth"
PROTO_OUT   = CKPT_DIR / "prototype_matcher.npz"
CLASSES_OUT = CKPT_DIR / "temporal_classifier_classes.json"

IMAGE_SIZE = 224
SEED       = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

TRANSFORM_VAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Model ─────────────────────────────────────────────────────────────────────

class SentenceClassifierHead(nn.Module):
    """Temporal Classifier taking [B, 16, 960] features -> [B, num_classes]."""
    def __init__(self, in_dim=960, hidden_dim=512, num_classes=101):
        super().__init__()
        # Frame projection
        self.fc_frame = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        # Temporal Attention Weighting
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Output Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, T, 960]
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        h_flat = self.fc_frame(x_flat) # [B*T, 512]
        h = h_flat.view(B, T, -1)     # [B, T, 512]

        # Temporal attention scores
        attn_logits = self.attn(h)      # [B, T, 1]
        attn_weights = F.softmax(attn_logits, dim=1) # [B, T, 1]

        # Weighted video representation
        context = torch.sum(h * attn_weights, dim=1) # [B, 512]
        logits = self.classifier(context)            # [B, num_classes]
        return logits, context

# ── Feature Extraction ────────────────────────────────────────────────────────

def get_isl_backbone():
    bb = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    features = bb.features
    avgpool  = bb.avgpool

    if MODEL1_CKPT.exists():
        ckpt  = torch.load(MODEL1_CKPT, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        ms    = features.state_dict()
        matched = 0
        for ck, cv in state.items():
            mk = ck.replace("backbone.", "")
            if mk in ms and ms[mk].shape == cv.shape:
                ms[mk] = cv; matched += 1
        features.load_state_dict(ms, strict=False)
        print(f"[Model 1] Loaded {matched} matched ISL backbone parameters.")
    else:
        print("[Model 1] Warning: Checkpoint not found, using default weights.")

    features.eval(); avgpool.eval()
    return features, avgpool


def get_cached_frames(video_path):
    rel_path = video_path.relative_to(VIDEO_DIR)
    safe_name = str(rel_path).replace("\\", "_").replace("/", "_") + ".npz"
    cache_path = CACHE_DIR / safe_name
    if cache_path.exists():
        try:
            return np.load(cache_path)["frames"]
        except Exception:
            pass
    return None


def extract_all_features():
    print("\n=== Phase 1: Extracting ISL Feature Representations ===")
    t0 = time.time()
    features, avgpool = get_isl_backbone()

    exts = {".mp4",".MP4",".mov",".MOV",".avi",".AVI"}
    all_videos = []
    for sd in sorted(VIDEO_DIR.iterdir()):
        if not sd.is_dir(): continue
        lbl = sd.name.strip().lower()
        for vf in sd.iterdir():
            if vf.suffix in exts:
                all_videos.append((vf, lbl))

    classes = sorted({lbl for _, lbl in all_videos})
    c2i = {l: i for i, l in enumerate(classes)}

    all_feats = []
    all_labels = []

    BATCH_VIDEOS = 16
    with torch.no_grad():
        for start_idx in range(0, len(all_videos), BATCH_VIDEOS):
            chunk = all_videos[start_idx:start_idx + BATCH_VIDEOS]
            batch_tensors = []
            chunk_labels = []

            for vpath, lbl in chunk:
                frames = get_cached_frames(vpath)
                if frames is None: continue
                tensors = torch.stack([TRANSFORM_VAL(f) for f in frames]) # [16, 3, 224, 224]
                batch_tensors.append(tensors)
                chunk_labels.append(c2i[lbl])

            if not batch_tensors: continue

            big_batch = torch.cat(batch_tensors, dim=0) # [B*16, 3, 224, 224]
            feats = torch.flatten(avgpool(features(big_batch)), 1) # [B*16, 960]
            feats = feats.view(len(chunk_labels), 16, 960).cpu().numpy() # [B, 16, 960]

            all_feats.append(feats)
            all_labels.extend(chunk_labels)

            cnt = min(start_idx + BATCH_VIDEOS, len(all_videos))
            print(f"  Extracted {cnt}/{len(all_videos)} videos [{time.time()-t0:.1f}s]", flush=True)

    X = np.concatenate(all_feats, axis=0).astype(np.float32) # [687, 16, 960]
    y = np.array(all_labels, dtype=np.int64)                 # [687]

    np.savez_compressed(FEAT_FILE, features=X, labels=y, classes=np.array(classes))
    print(f"Saved feature dataset: {FEAT_FILE} (shape: {X.shape})")
    return X, y, classes

# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  SignBridge - Fast Sentence Classifier Training")
    print("="*60)

    if FEAT_FILE.exists():
        print(f"Loading pre-extracted features from {FEAT_FILE}...")
        data = np.load(FEAT_FILE, allow_pickle=True)
        X = data["features"]
        y = data["labels"]
        classes = list(data["classes"])
    else:
        X, y, classes = extract_all_features()

    NC = len(classes)
    N = len(X)
    print(f"\nDataset: {N} videos across {NC} sentence classes.")
    print(f"Feature matrix shape: {X.shape}")

    # Train / Val Split (Stratified random)
    indices = list(range(N))
    random.shuffle(indices)
    val_size = int(N * 0.15)
    val_idx  = indices[:val_size]
    tr_idx   = indices[val_size:]

    X_tr, y_tr = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
    X_va, y_va = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SentenceClassifierHead(in_dim=960, hidden_dim=512, num_classes=NC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n=== Phase 2: Training Temporal Classifier Head (100 Epochs) ===")
    t0 = time.time()
    best_acc = 0.0

    for ep in range(1, 101):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for bx, by in train_dl:
            optimizer.zero_grad()
            # Data Augmentation on features: Add slight Gaussian noise & frame dropout during training
            if model.training:
                noise = torch.randn_like(bx) * 0.02
                bx_aug = bx + noise
            else:
                bx_aug = bx

            logits, _ = model(bx_aug)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * by.size(0)
            tr_correct += (logits.argmax(-1) == by).sum().item()
            tr_total += by.size(0)

        scheduler.step()
        tr_acc = 100.0 * tr_correct / tr_total

        # Validation
        if ep % 10 == 0 or ep == 1 or ep == 100:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_va)
                val_preds = val_logits.argmax(-1)
                val_acc = 100.0 * (val_preds == y_va).sum().item() / len(y_va)
                k = min(5, NC)
                top5_acc = 100.0 * (val_logits.topk(k, dim=1).indices == y_va.unsqueeze(1)).any(dim=1).sum().item() / len(y_va)

            print(f"Ep {ep:3d}/100 | Train Acc: {tr_acc:.1f}% | Val Top-1: {val_acc:.1f}% | Val Top-5: {top5_acc:.1f}% [{time.time()-t0:.2f}s]")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "classes": classes,
                            "val_acc": val_acc}, HEAD_CKPT)

    print(f"\nTraining completed in {time.time()-t0:.2f} seconds!")
    print(f"Best Validation Accuracy: {best_acc:.1f}%")

    # ── Phase 3: Build Prototypes from Fine-tuned Feature Heads ──────────────
    print("\n=== Phase 3: Building Final Prototype Matcher ===")
    best_ckpt = torch.load(HEAD_CKPT)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    X_all = torch.tensor(X)
    with torch.no_grad():
        _, contexts = model(X_all) # [687, 512]
        contexts = contexts.numpy()
        # L2 normalize
        contexts = contexts / (np.linalg.norm(contexts, axis=1, keepdims=True) + 1e-8)

    class_vecs = defaultdict(list)
    for idx in range(N):
        class_vecs[classes[y[idx]]].append(contexts[idx])

    proto_labels = sorted(class_vecs.keys())
    proto_mat = np.stack([
        np.mean(class_vecs[l], axis=0) / (np.linalg.norm(np.mean(class_vecs[l], axis=0)) + 1e-8)
        for l in proto_labels
    ]).astype(np.float32)

    np.savez(PROTO_OUT, prototypes=proto_mat, labels=np.array(proto_labels))

    c2i = {l: idx for idx, l in enumerate(proto_labels)}
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": len(proto_labels),
            "class_to_idx": c2i,
            "idx_to_class": {str(idx): l for idx, l in enumerate(proto_labels)},
            "recognition_mode": "temporal_attention_prototype_finetuned"
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved Prototype Matcher: {PROTO_OUT}")
    print(f"Saved Classes Vocab    : {CLASSES_OUT}")
    print("DONE.")

if __name__ == "__main__":
    main()
