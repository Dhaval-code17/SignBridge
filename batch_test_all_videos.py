"""
batch_test_all_videos.py
=========================
Tests ALL 687 videos from the ISL_CSLRT_Corpus dataset
and shows which ones predict correctly (Top-1 and Top-5).
"""

import numpy as np, torch
from pathlib import Path
from torchvision import models, transforms
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from training.train_fast_sentence_classifier import SentenceClassifierHead

ROOT      = Path(__file__).resolve().parent
VIDEO_DIR = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
CACHE_DIR = ROOT / "data" / "sentence_frames_cache"
CKPT_DIR  = ROOT / "models" / "checkpoints"
PROTO_OUT = CKPT_DIR / "prototype_matcher.npz"
HEAD_CKPT = CKPT_DIR / "sentence_classifier_head.pth"
MODEL1_CKPT = CKPT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"

IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def main():
    print("="*60)
    print("  Batch Testing All 687 ISL Sentence Videos")
    print("="*60)

    # Load backbone
    bb = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    features = bb.features
    avgpool  = bb.avgpool
    ckpt  = torch.load(str(MODEL1_CKPT), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ms = features.state_dict()
    matched = 0
    for ck, cv in state.items():
        mk = ck.replace("backbone.", "")
        if mk in ms and ms[mk].shape == cv.shape:
            ms[mk] = cv; matched += 1
    features.load_state_dict(ms, strict=False)
    features.eval(); avgpool.eval()
    print(f"[Backbone] Loaded {matched} ISL params")

    # Load prototypes + head
    pdata = np.load(str(PROTO_OUT))
    prototypes = pdata["prototypes"]  # [101, 512]
    labels = list(pdata["labels"])

    h_ckpt = torch.load(str(HEAD_CKPT), map_location="cpu")
    head = SentenceClassifierHead(in_dim=960, hidden_dim=512, num_classes=len(labels))
    head.load_state_dict(h_ckpt["model_state_dict"]); head.eval()
    print(f"[Head] Loaded. Prototypes: {prototypes.shape}")

    exts = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}

    def get_cached(vpath):
        rel = vpath.relative_to(VIDEO_DIR)
        safe = str(rel).replace("\\", "_").replace("/", "_") + ".npz"
        cp = CACHE_DIR / safe
        if cp.exists():
            try: return np.load(str(cp))["frames"]
            except: pass
        return None

    correct_top1 = []
    correct_top5 = []
    wrong        = []
    all_videos   = []

    for sd in sorted(VIDEO_DIR.iterdir()):
        if not sd.is_dir(): continue
        true_lbl = sd.name.strip().lower()
        for vf in sorted(sd.iterdir()):
            if vf.suffix not in exts: continue
            all_videos.append((vf, true_lbl))

    print(f"\nTesting {len(all_videos)} videos...\n")

    for vf, true_lbl in all_videos:
        frames = get_cached(vf)
        if frames is None:
            wrong.append((vf.parent.name, vf.name, "NO_CACHE", true_lbl))
            continue
        tensors = torch.stack([TRANSFORM(f) for f in frames])
        with torch.no_grad():
            feats = torch.flatten(avgpool(features(tensors)), 1)
            _, ctx = head(feats.unsqueeze(0))
            q = ctx.squeeze(0).numpy()
            q /= (np.linalg.norm(q) + 1e-8)

        sims = np.dot(prototypes, q)
        top5_idx = np.argsort(sims)[::-1][:5]
        top1_lbl = labels[top5_idx[0]]
        top5_lbls = [labels[i] for i in top5_idx]

        if top1_lbl == true_lbl:
            correct_top1.append((vf.parent.name, vf.name, top1_lbl, float(sims[top5_idx[0]])))
            correct_top5.append((vf.parent.name, vf.name, top1_lbl, float(sims[top5_idx[0]])))
        elif true_lbl in top5_lbls:
            correct_top5.append((vf.parent.name, vf.name, top1_lbl, float(sims[top5_idx[0]])))
            wrong.append((vf.parent.name, vf.name, top1_lbl, true_lbl))
        else:
            wrong.append((vf.parent.name, vf.name, top1_lbl, true_lbl))

    total = len(all_videos)
    print(f"Total videos tested : {total}")
    print(f"Top-1 Correct       : {len(correct_top1)} ({100*len(correct_top1)/total:.1f}%)")
    print(f"Top-5 Correct       : {len(correct_top5)} ({100*len(correct_top5)/total:.1f}%)")
    print(f"Wrong (not in Top-5): {total - len(correct_top5)} ({100*(total-len(correct_top5))/total:.1f}%)")

    print("\n" + "="*60)
    print("VIDEOS PREDICTED CORRECTLY (Top-1):")
    print("="*60)
    for folder, fname, pred, sim in sorted(correct_top1, key=lambda x: x[0]):
        print(f"  [OK]  {folder}/{fname}")
        print(f"        -> {pred}  (sim={sim:.3f})")

    print("\n" + "="*60)
    print("VIDEOS PREDICTED WRONG (predicted: X | actual: Y):")
    print("="*60)
    for folder, fname, pred, true in sorted(wrong, key=lambda x: x[0]):
        print(f"  [WRONG] {folder}/{fname}")
        print(f"          predicted: {pred}  |  actual: {true}")

if __name__ == "__main__":
    main()
