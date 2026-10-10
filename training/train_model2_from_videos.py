"""
train_model2_from_videos.py
===========================
Trains Model 2 (Temporal Transformer) for SignBridge.

Step 1  Extract [T, 960] visual features from ISL sentence-level videos
        using the real trained Model 1 (MobileNetV3) checkpoint.
        Features are cached to data/model2_sentence_features/ for speed.

Step 2  Train Temporal Transformer classifier (Model 2) on those features.

Step 3  Save:
          models/checkpoints/temporal_classifier_best.pth
          models/checkpoints/temporal_classifier_classes.json
"""

from __future__ import annotations

import json, os, random, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parents[1]
VIDEO_DIR     = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
CKPT_DIR      = ROOT / "models" / "checkpoints"
MODEL1_CKPT   = CKPT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"
MODEL2_OUT    = CKPT_DIR / "temporal_classifier_best.pth"
CLASSES_OUT   = CKPT_DIR / "temporal_classifier_classes.json"
FEAT_CACHE    = ROOT / "data" / "model2_sentence_features"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_CACHE.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────

IMAGE_SIZE   = 224
SAMPLE_FPS   = 8.0
MAX_FRAMES   = 120
INPUT_DIM    = 960
D_MODEL      = 256
NHEAD        = 8
NUM_LAYERS   = 4
DIM_FF       = 512
DROPOUT      = 0.2
EPOCHS       = 60
BATCH_SIZE   = 16
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
LABEL_SMOOTH = 0.1
VAL_SPLIT    = 0.15
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Model 1 (visual encoder) ─────────────────────────────────────────────────

class Model1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.mobilenet_v3_large(weights=None)
        self.features = bb.features
        self.avgpool  = bb.avgpool
    def forward(self, x):
        return torch.flatten(self.avgpool(self.features(x)), 1)

def load_model1():
    m = Model1Encoder()
    if MODEL1_CKPT.exists():
        ckpt  = torch.load(MODEL1_CKPT, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        ms    = m.state_dict()
        ms.update({k:v for k,v in state.items() if k in ms})
        m.load_state_dict(ms)
        print(f"[Model 1] Loaded checkpoint  ({MODEL1_CKPT.name})")
    else:
        print("[Model 1] Checkpoint missing – using ImageNet weights")
        bb = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.features = bb.features; m.avgpool = bb.avgpool
    return m.to(DEVICE).eval()

# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur    = nf / fps
    target = max(4, min(MAX_FRAMES, int(round(dur * SAMPLE_FPS))))
    wanted = set(np.linspace(0, nf-1, target).astype(int).tolist())
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if idx in wanted:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames

# ── Feature extraction with cache ────────────────────────────────────────────

def get_features(model1, video_path):
    safe_name = video_path.stem[:40] + "_" + video_path.parent.name[:20].replace(" ","_")
    cache = FEAT_CACHE / (safe_name + ".npy")
    if cache.exists():
        return np.load(cache)
    frames = extract_frames(video_path)
    if len(frames) < 2:
        return None
    tensors = [TRANSFORM(f) for f in frames]
    parts = []
    with torch.inference_mode():
        for i in range(0, len(tensors), 16):
            batch = torch.stack(tensors[i:i+16]).to(DEVICE)
            parts.append(model1(batch).cpu().numpy())
    feat = np.concatenate(parts, 0).astype(np.float32)
    np.save(cache, feat)
    return feat

# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset(model1):
    print("\n=== STEP 1: Feature Extraction ===")
    exts = {".mp4",".MP4",".mov",".MOV",".avi",".AVI"}
    items, skipped = [], 0
    sent_dirs = sorted(d for d in VIDEO_DIR.iterdir() if d.is_dir())
    print(f"Sentence categories: {len(sent_dirs)}", flush=True)
    for idx, sd in enumerate(sent_dirs, 1):
        label = sd.name.strip().lower()
        cat_count = 0
        for vf in sd.iterdir():
            if vf.suffix not in exts:
                continue
            feat = get_features(model1, vf)
            if feat is None or feat.shape[0] < 2:
                skipped += 1
                continue
            items.append((feat, label))
            cat_count += 1
        print(f"  [{idx:3d}/{len(sent_dirs)}] {label[:40]:<40} +{cat_count} videos  (total={len(items)})", flush=True)
    print(f"Videos processed : {len(items)}  (skipped {skipped})", flush=True)
    return items

# ── Dataset / collate ────────────────────────────────────────────────────────

class SentenceDS(Dataset):
    def __init__(self, items, c2i):
        self.items = items; self.c2i = c2i
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        feat, lbl = self.items[i]
        return feat, self.c2i[lbl]

def collate(batch):
    feats, lbls = zip(*batch)
    lens    = [f.shape[0] for f in feats]
    maxlen  = max(lens)
    B, D    = len(feats), feats[0].shape[1]
    padded  = torch.zeros(B, maxlen, D)
    mask    = torch.ones(B, maxlen, dtype=torch.bool)
    for i,(f,l) in enumerate(zip(feats,lens)):
        padded[i,:l] = torch.tensor(f); mask[i,:l] = False
    return padded, mask, torch.tensor(lbls, dtype=torch.long)

# ── Model 2 (Temporal Transformer) ───────────────────────────────────────────

class TemporalTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD,
            dim_feedforward=DIM_FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, NUM_LAYERS)
        self.head    = nn.Linear(D_MODEL, num_classes)

    def _pe(self, x):
        B,T,D = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(1).float()
        div  = torch.exp(torch.arange(0,D,2,device=x.device).float()*(-np.log(10000.0)/D))
        pe   = torch.zeros(T,D,device=x.device)
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div[:D//2])
        return x + pe.unsqueeze(0)

    def forward(self, x, mask=None):
        x = self.drop(self._pe(self.proj(x)))
        x = self.encoder(x, src_key_padding_mask=mask)
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            x = (x*valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x)

# ── Loss ─────────────────────────────────────────────────────────────────────

class LabelSmoothCE(nn.Module):
    def __init__(self, s=0.1): super().__init__(); self.s = s
    def forward(self, logits, targets):
        lp  = F.log_softmax(logits, -1)
        nll = -lp.gather(-1, targets.unsqueeze(1)).squeeze(1)
        return ((1-self.s)*nll + self.s*(-lp.mean(-1))).mean()

# ── Train / eval loops ───────────────────────────────────────────────────────

def train_epoch(model, loader, crit, opt, scaler):
    model.train(); tloss=correct=total=0
    for feats,masks,lbls in loader:
        feats=feats.to(DEVICE); masks=masks.to(DEVICE); lbls=lbls.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            out  = model(feats, masks)
            loss = crit(out, lbls)
        if DEVICE.type=="cuda":
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        tloss+=loss.item()*lbls.size(0); correct+=(out.argmax(-1)==lbls).sum().item(); total+=lbls.size(0)
    return tloss/total, 100.*correct/total

@torch.no_grad()
def eval_epoch(model, loader, crit):
    model.eval(); tloss=correct=top5=total=0
    for feats,masks,lbls in loader:
        feats=feats.to(DEVICE); masks=masks.to(DEVICE); lbls=lbls.to(DEVICE)
        out  = model(feats, masks)
        loss = crit(out, lbls)
        tloss+=loss.item()*lbls.size(0)
        correct+=(out.argmax(-1)==lbls).sum().item()
        top5+=(out.topk(min(5,out.size(-1)),-1).indices==lbls.unsqueeze(1)).any(1).sum().item()
        total+=lbls.size(0)
    return tloss/total, 100.*correct/total, 100.*top5/total

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  SignBridge — Model 2 Training")
    print(f"  Device : {DEVICE}   Epochs : {EPOCHS}")
    print("="*60)

    model1   = load_model1()
    all_items = build_dataset(model1)

    if not all_items:
        raise RuntimeError(f"No videos found in {VIDEO_DIR}")

    all_labels  = sorted({lbl for _,lbl in all_items})
    c2i = {l:i for i,l in enumerate(all_labels)}
    i2c = {i:l for l,i in c2i.items()}
    NC  = len(all_labels)
    print(f"\nClasses : {NC}")

    random.shuffle(all_items)
    nval = max(1, int(len(all_items)*VAL_SPLIT))
    val_items, tr_items = all_items[:nval], all_items[nval:]
    print(f"Train   : {len(tr_items)}   Val : {len(val_items)}")

    tr_dl = DataLoader(SentenceDS(tr_items,c2i), BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=0)
    va_dl = DataLoader(SentenceDS(val_items,c2i),BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

    model2  = TemporalTransformer(NC).to(DEVICE)
    crit    = LabelSmoothCE(LABEL_SMOOTH)
    opt     = torch.optim.AdamW(model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.01)
    scaler  = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")

    print("\n=== STEP 2: Training ===")
    best=0.0; t0=time.time()
    for ep in range(1, EPOCHS+1):
        tl,ta = train_epoch(model2, tr_dl, crit, opt, scaler)
        vl,va,v5 = eval_epoch(model2, va_dl, crit)
        sched.step()
        print(f"Ep {ep:3d}/{EPOCHS}  tr={ta:.1f}%  val={va:.1f}%  top5={v5:.1f}%  [{time.time()-t0:.0f}s]")
        if va > best:
            best = va
            torch.save({"epoch":ep,"model_state_dict":model2.state_dict(),
                        "val_acc":va,"val_top5":v5,"num_classes":NC,
                        "config":{"input_dim":INPUT_DIM,"d_model":D_MODEL,"nhead":NHEAD,
                                  "num_layers":NUM_LAYERS,"dim_feedforward":DIM_FF,"dropout":DROPOUT}},
                       MODEL2_OUT)
            print(f"  [SAVED] best val={va:.1f}%", flush=True)

    with open(CLASSES_OUT,"w",encoding="utf-8") as f:
        json.dump({"num_classes":NC,"class_to_idx":c2i,"idx_to_class":{str(k):v for k,v in i2c.items()}},f,indent=2,ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DONE - best val acc: {best:.1f}%")
    print(f"Checkpoint : {MODEL2_OUT}")
    print(f"Classes    : {CLASSES_OUT}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
