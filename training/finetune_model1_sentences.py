"""
finetune_model1_sentences.py  (fast pre-cached frame version)
=============================================================
Fine-tunes Model 1 (MobileNetV3) on 101 ISL sentence classes.
Pre-caches extracted video frames to disk once for 100x faster CPU training.
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

# ── Paths ─────────────────────────────────────────────────────────────────────

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

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_SIZE       = 224
FRAMES_PER_VIDEO = 16    # 16 frames per video cached
FRAMES_PER_CLIP  = 8     # random 8 frames sampled per training iteration
EPOCHS           = 30
BATCH_SIZE       = 8     # video batch size
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

# ── Model ─────────────────────────────────────────────────────────────────────

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
        feat = torch.flatten(self.avgpool(self.features(x)), 1)   # [B, 960]
        return self.head(feat), feat


def load_model(num_classes):
    model = SentenceClassifier(num_classes)

    # Load Model 1 backbone weights if available
    if MODEL1_CKPT.exists():
        ckpt  = torch.load(MODEL1_CKPT, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        ms    = model.state_dict()
        
        # Remap backbone. -> features.
        matched = 0
        for ck, cv in state.items():
            mk = ck.replace("backbone.", "features.")
            if mk in ms and ms[mk].shape == cv.shape:
                ms[mk] = cv
                matched += 1
        model.load_state_dict(ms, strict=False)
        print(f"[Model 1] Loaded {matched} matched ISL backbone params from checkpoint")
    else:
        print("[Model 1] Using ImageNet pretrained weights (no ISL checkpoint)")

    # Fine-tune last 4 blocks + classifier head
    for i, layer in enumerate(model.features):
        for p in layer.parameters():
            p.requires_grad = (i >= 12)
    for p in model.head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params")
    return model.to(DEVICE)

# ── Frame Caching Utility ────────────────────────────────────────────────────

def get_cached_frames(video_path):
    """Extract FRAMES_PER_VIDEO frames from video and cache as npz array."""
    rel_path = video_path.relative_to(VIDEO_DIR)
    safe_name = str(rel_path).replace("\\", "_").replace("/", "_") + ".npz"
    cache_path = CACHE_DIR / safe_name
    
    if cache_path.exists():
        try:
            return np.load(cache_path)["frames"]
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((FRAMES_PER_VIDEO, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = set(np.linspace(0, total-1, min(FRAMES_PER_VIDEO, total)).astype(int).tolist())
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if idx in indices:
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
            frames.append(f)
        idx += 1
    cap.release()

    if not frames:
        frames = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)]
    
    # Pad if fewer than FRAMES_PER_VIDEO
    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(frames[-1])

    frames_arr = np.array(frames[:FRAMES_PER_VIDEO], dtype=np.uint8)
    np.savez_compressed(cache_path, frames=frames_arr)
    return frames_arr

# ── Cached Dataset ───────────────────────────────────────────────────────────

class FastCachedDataset(Dataset):
    def __init__(self, video_items, transform, is_train=True):
        self.items = video_items  # list of (Path, label_idx)
        self.transform = transform
        self.is_train = is_train

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        vpath, lbl = self.items[idx]
        frames = get_cached_frames(vpath)  # [16, 224, 224, 3] uint8
        
        if self.is_train:
            # Sample FRAMES_PER_CLIP frames randomly
            indices = np.random.choice(len(frames), FRAMES_PER_CLIP, replace=False)
            sampled = frames[indices]
        else:
            sampled = frames

        tensors = [self.transform(f) for f in sampled]
        return torch.stack(tensors), lbl


def flat_collate(batch):
    all_frames, all_labels = [], []
    for frames_tensor, lbl in batch:
        all_frames.append(frames_tensor)
        all_labels.extend([lbl] * frames_tensor.shape[0])
    return torch.cat(all_frames, 0), torch.tensor(all_labels, dtype=torch.long)

# ── Eval (video-level) ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_video_acc(model, val_items, num_classes):
    model.eval(); correct = top5 = total = 0
    for vpath, true_idx in val_items:
        frames = get_cached_frames(vpath)  # [16, 224, 224, 3]
        tensors = [TRANSFORM_VAL(f) for f in frames]
        batch = torch.stack(tensors).to(DEVICE)
        logits, _ = model(batch)
        mean_logit = logits.mean(0)
        pred = int(mean_logit.argmax())
        correct += (pred == true_idx)
        k = min(5, num_classes)
        top5 += bool((mean_logit.topk(k).indices == true_idx).any())
        total += 1
    acc  = 100. * correct / max(total, 1)
    acc5 = 100. * top5    / max(total, 1)
    return acc, acc5

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  SignBridge - Fine-tune Model 1 on 101 Sentences (Fast Cached)")
    print(f"  Device: {DEVICE}  Epochs: {EPOCHS}")
    print("="*60)

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
    i2c = {i:l for l,i in c2i.items()}
    NC  = len(classes)

    all_idx = [(vf, c2i[lbl]) for vf, lbl in all_items]
    random.shuffle(all_idx)
    nval      = max(NC, int(len(all_idx) * VAL_SPLIT))
    val_items = all_idx[:nval]
    tr_items  = all_idx[nval:]
    print(f"Classes: {NC}  Train: {len(tr_items)}  Val: {len(val_items)}")

    print("\n=== Pre-caching video frames to disk ===")
    t_cache = time.time()
    for i, (vf, _) in enumerate(all_items, 1):
        get_cached_frames(vf)
        if i % 100 == 0 or i == len(all_items):
            print(f"  Cached {i}/{len(all_items)} videos [{time.time()-t_cache:.1f}s]", flush=True)

    model  = load_model(NC)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")

    backbone_params = [p for n,p in model.named_parameters() if "head" not in n and p.requires_grad]
    head_params     = [p for n,p in model.named_parameters() if "head"     in n and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params,     "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_HEAD*0.01)
    crit  = nn.CrossEntropyLoss()

    tr_ds = FastCachedDataset(tr_items, TRANSFORM_TRAIN, is_train=True)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=flat_collate, num_workers=0)

    print("\n=== Training ===")
    best_acc = 0.0; t0 = time.time()

    for ep in range(1, EPOCHS+1):
        model.train(); loss_sum = correct = total = 0
        for frames, lbls in tr_dl:
            frames = frames.to(DEVICE); lbls = lbls.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
                logits, _ = model(frames)
                loss = crit(logits, lbls)
            if DEVICE.type=="cuda":
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            loss_sum += loss.item()*lbls.size(0)
            correct  += (logits.argmax(-1)==lbls).sum().item()
            total    += lbls.size(0)
        sched.step()
        tr_acc = 100.*correct/total

        if ep % 3 == 0 or ep == 1 or ep == EPOCHS:
            va, v5 = eval_video_acc(model, val_items, NC)
            print(f"Ep {ep:2d}/{EPOCHS}  tr={tr_acc:.1f}%  val={va:.1f}%  top5={v5:.1f}%  [{time.time()-t0:.0f}s]", flush=True)
            if va > best_acc:
                best_acc = va
                torch.save({"epoch":ep,"model_state_dict":model.state_dict(),
                            "val_acc":va,"num_classes":NC,
                            "c2i":c2i,"i2c":{str(k):v for k,v in i2c.items()}},
                           SENT_CKPT)
                print(f"  [SAVED] best val={va:.1f}%", flush=True)
        else:
            print(f"Ep {ep:2d}/{EPOCHS}  tr={tr_acc:.1f}%  [{time.time()-t0:.0f}s]", flush=True)

    # ── Build final prototype matcher using fine-tuned features ───────────────
    print("\n=== Building prototype matcher with fine-tuned features ===")
    best = torch.load(SENT_CKPT, map_location=DEVICE)
    model.load_state_dict(best["model_state_dict"]); model.eval()

    label_to_vecs = defaultdict(list)
    for sd in sorted(VIDEO_DIR.iterdir()):
        if not sd.is_dir(): continue
        lbl = sd.name.strip().lower()
        for vf in sd.iterdir():
            if vf.suffix not in exts: continue
            frames = get_cached_frames(vf)
            tensors = [TRANSFORM_VAL(f) for f in frames]
            with torch.no_grad():
                batch = torch.stack(tensors).to(DEVICE)
                _, feat = model(batch)
                vec = feat.mean(0).cpu().numpy()
                vec /= (np.linalg.norm(vec)+1e-8)
            label_to_vecs[lbl].append(vec)

    proto_labels = sorted(label_to_vecs.keys())
    proto_mat = np.stack([
        np.mean(label_to_vecs[l], 0) / (np.linalg.norm(np.mean(label_to_vecs[l],0))+1e-8)
        for l in proto_labels
    ]).astype(np.float32)

    np.savez(PROTO_OUT, prototypes=proto_mat, labels=np.array(proto_labels))
    fc2i = {l:i for i,l in enumerate(proto_labels)}
    with open(CLASSES_OUT,"w",encoding="utf-8") as f:
        json.dump({"num_classes":len(proto_labels),"class_to_idx":fc2i,
                   "idx_to_class":{str(i):l for l,i in fc2i.items()},
                   "recognition_mode":"prototype_cosine_finetuned"},
                  f,indent=2,ensure_ascii=False)

    print(f"Fine-tuned model : {SENT_CKPT}")
    print(f"Prototype matcher: {PROTO_OUT}")
    print(f"Classes file     : {CLASSES_OUT}")
    print(f"Best val acc     : {best_acc:.1f}%")
    print("DONE.")

if __name__ == "__main__":
    main()

