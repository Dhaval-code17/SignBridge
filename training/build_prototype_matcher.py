"""
build_prototype_matcher.py
==========================
Builds a prototype-based matcher for sentence-level ISL recognition.

Instead of a neural classifier, this:
1. Computes a mean-pooled feature vector for every video
2. Averages those across all videos for each class -> class prototype
3. At test time: cosine-similarity to all prototypes -> top-1 prediction

Works far better than a neural classifier when you have <10 samples/class.
Saves: models/checkpoints/temporal_classifier_best.pth  (same path, compatible)
       models/checkpoints/temporal_classifier_classes.json
"""

import json, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parents[1]
VIDEO_DIR  = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
FEAT_CACHE = ROOT / "data" / "model2_sentence_features"
CKPT_DIR   = ROOT / "models" / "checkpoints"
PROTO_OUT  = CKPT_DIR / "prototype_matcher.npz"
CLASSES_OUT= CKPT_DIR / "temporal_classifier_classes.json"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load feature cache for every video ───────────────────────────────────────

exts = {".mp4",".MP4",".mov",".MOV",".avi",".AVI"}

label_to_feats = defaultdict(list)
skipped = 0

print("Loading cached features...")
for sd in sorted(VIDEO_DIR.iterdir()):
    if not sd.is_dir(): continue
    label = sd.name.strip().lower()
    for vf in sd.iterdir():
        if vf.suffix not in exts: continue
        safe  = vf.stem[:40] + "_" + sd.name[:20].replace(" ","_")
        cache = FEAT_CACHE / (safe + ".npy")
        if not cache.exists():
            skipped += 1
            continue
        feat = np.load(cache)         # [T, 960]
        vec  = feat.mean(axis=0)      # [960]  mean-pool over time
        vec  = vec / (np.linalg.norm(vec) + 1e-8)   # L2-normalise
        label_to_feats[label].append(vec)

classes = sorted(label_to_feats.keys())
print(f"Classes loaded  : {len(classes)}")
print(f"Total videos    : {sum(len(v) for v in label_to_feats.values())}")
print(f"Skipped (no cache): {skipped}")

# ── Leave-one-out evaluation ─────────────────────────────────────────────────

print("\nLeave-one-out evaluation...")
top1 = top5 = total = 0

for test_label in classes:
    test_vecs = label_to_feats[test_label]
    for i, test_vec in enumerate(test_vecs):
        # Build prototypes excluding this test video
        proto_vecs = []
        proto_labels = []
        for lbl in classes:
            vecs = label_to_feats[lbl]
            if lbl == test_label:
                # exclude the current test sample
                others = [v for j,v in enumerate(vecs) if j != i]
            else:
                others = vecs
            if not others:
                continue
            proto = np.mean(others, axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            proto_vecs.append(proto)
            proto_labels.append(lbl)

        proto_mat = np.stack(proto_vecs)          # [C, 960]
        sims      = proto_mat @ test_vec          # [C]
        ranked    = np.argsort(-sims)

        top1 += (proto_labels[ranked[0]] == test_label)
        top5 += any(proto_labels[ranked[j]] == test_label for j in range(min(5, len(proto_labels))))
        total += 1

print(f"LOO Top-1 accuracy : {100.*top1/total:.1f}%  ({top1}/{total})")
print(f"LOO Top-5 accuracy : {100.*top5/total:.1f}%  ({top5}/{total})")

# ── Build FINAL prototypes (all videos included) ──────────────────────────────

print("\nBuilding final prototypes (all videos)...")
proto_matrix = []
proto_labels = []

for lbl in classes:
    vecs  = label_to_feats[lbl]
    proto = np.mean(vecs, axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    proto_matrix.append(proto)
    proto_labels.append(lbl)

proto_matrix = np.stack(proto_matrix).astype(np.float32)   # [101, 960]
print(f"Prototype matrix shape: {proto_matrix.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────

np.savez(PROTO_OUT, prototypes=proto_matrix, labels=np.array(proto_labels))

class_to_idx = {lbl: i for i, lbl in enumerate(classes)}
idx_to_class = {str(i): lbl for i, lbl in enumerate(classes)}

with open(CLASSES_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "num_classes":  len(classes),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "recognition_mode": "prototype_cosine",
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved prototype matcher : {PROTO_OUT}")
print(f"Saved class vocabulary  : {CLASSES_OUT}")
print("\nDONE.")
