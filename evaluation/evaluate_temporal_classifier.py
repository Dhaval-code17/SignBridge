"""
Person 2 - Model 2 Evaluation Script
=====================================
Evaluates the trained TemporalTransformerClassifier.
Reports:
  - Top-1 accuracy
  - Top-5 accuracy
  - Per-class accuracy (for classes with support)
  - Sample predictions

Usage:
    python evaluation/evaluate_temporal_classifier.py
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_temporal_classifier import (
    TemporalClassifierDataset,
    TemporalTransformerClassifier,
    collate_fn,
    build_vocab,
)

# ============================================================
# Paths
# ============================================================

ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_DIR         = os.path.join(ROOT, "data", "continuous", "new_val")
TRAIN_DIR       = os.path.join(ROOT, "data", "continuous", "new_train")
CHECKPOINT_DIR  = os.path.join(ROOT, "models", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "temporal_classifier_best.pth")
CLASSES_PATH    = os.path.join(CHECKPOINT_DIR, "temporal_classifier_classes.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ============================================================
# Load vocabulary
# ============================================================

print("Loading class vocabulary...")
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    class_to_idx = classes_data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in classes_data["idx_to_class"].items()}
    num_classes  = classes_data["num_classes"]
else:
    print("Classes file not found, building from dataset...")
    class_to_idx, idx_to_class = build_vocab(TRAIN_DIR, VAL_DIR)
    num_classes = len(class_to_idx)

print(f"Classes: {num_classes}")

# ============================================================
# Load checkpoint and config
# ============================================================

print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
config = checkpoint.get("config", {
    "input_dim": 960, "d_model": 256, "nhead": 8,
    "num_layers": 4, "dim_feedforward": 512,
    "dropout": 0.2, "num_classes": num_classes,
})

model = TemporalTransformerClassifier(**config).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
print(f"Saved val acc: {checkpoint.get('val_acc', 0) * 100:.2f}%")

# ============================================================
# Dataset & loader
# ============================================================

val_ds = TemporalClassifierDataset(VAL_DIR, class_to_idx, augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

print(f"\nValidation sequences: {len(val_ds)}")

# ============================================================
# Evaluation
# ============================================================

correct    = 0
top5       = 0
total      = 0
per_class_correct = {}
per_class_total   = {}

sample_preds = []

with torch.no_grad():
    for features, padding_mask, labels, _ in val_loader:
        features     = features.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)
        labels_dev   = labels.to(DEVICE)

        logits = model(features, padding_mask)
        preds  = logits.argmax(dim=-1)

        correct += (preds == labels_dev).sum().item()
        top5_pred = logits.topk(5, dim=-1).indices
        top5    += (top5_pred == labels_dev.unsqueeze(1)).any(dim=-1).sum().item()
        total   += labels.size(0)

        # Per-class tracking
        for true_lbl, pred_lbl in zip(labels.tolist(), preds.tolist()):
            cls_name = idx_to_class[true_lbl]
            per_class_total[cls_name]   = per_class_total.get(cls_name, 0) + 1
            per_class_correct[cls_name] = per_class_correct.get(cls_name, 0) + (true_lbl == pred_lbl)

        # Collect samples
        if len(sample_preds) < 15:
            for true_lbl, pred_lbl in zip(labels.tolist(), preds.tolist()):
                if len(sample_preds) < 15:
                    sample_preds.append({
                        "expected": idx_to_class[true_lbl],
                        "predicted": idx_to_class[pred_lbl],
                        "correct": true_lbl == pred_lbl,
                    })

# ============================================================
# Results
# ============================================================

top1_acc = correct / total
top5_acc = top5 / total

print()
print("=" * 60)
print("Person 2 — Temporal Transformer Evaluation Results")
print("=" * 60)
print(f"Validation sequences : {total}")
print(f"Top-1 accuracy       : {top1_acc * 100:.2f}%")
print(f"Top-5 accuracy       : {top5_acc * 100:.2f}%")
print("=" * 60)

# Sample predictions
print("\nSample predictions (first 15):")
print(f"  {'Expected':<30}  {'Predicted':<30}  {'OK?':>4}")
print("  " + "-" * 68)
for s in sample_preds:
    ok = "YES" if s["correct"] else "no"
    print(f"  {s['expected'][:28]:<30}  {s['predicted'][:28]:<30}  {ok:>4}")

# Top correct classes
correct_classes = {
    cls: per_class_correct.get(cls, 0) / per_class_total[cls]
    for cls in per_class_total
    if per_class_total[cls] >= 1
}
top_classes = sorted(correct_classes.items(), key=lambda x: -x[1])[:10]
print("\nTop 10 best-recognized classes:")
for cls, acc in top_classes:
    n = per_class_total[cls]
    print(f"  {cls:<35} {acc*100:>6.1f}%  (n={n})")

print("=" * 60)
print(f"\nTarget: 40-50% accuracy")
print(f"Achieved: {top1_acc*100:.2f}% top-1,  {top5_acc*100:.2f}% top-5")
if top1_acc >= 0.40:
    print("✓ TARGET ACHIEVED!")
else:
    print(f"  Gap to target: {(0.40 - top1_acc) * 100:.2f}% remaining")
print("=" * 60)
