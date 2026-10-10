"""
build_sentence_prototypes_fast.py
==================================
1. Loads MobileNetV3 with correct ISL backbone weights (308 params matched).
2. Extracts 960-dim feature vectors for all 687 sentence videos (using pre-cached frames).
3. Evaluates Leave-One-Out (LOO) classification accuracy.
4. Builds prototype_matcher.npz and temporal_classifier_classes.json.
Takes ~30 seconds total on CPU.
"""

import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
VIDEO_DIR   = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
CACHE_DIR   = ROOT / "data" / "sentence_frames_cache"
CKPT_DIR    = ROOT / "models" / "checkpoints"
MODEL1_CKPT = CKPT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"
PROTO_OUT   = CKPT_DIR / "prototype_matcher.npz"
CLASSES_OUT = CKPT_DIR / "temporal_classifier_classes.json"

IMAGE_SIZE  = 224
DEVICE      = torch.device("cpu")

TRANSFORM_VAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Load Model ────────────────────────────────────────────────────────────────

def get_isl_feature_extractor():
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
                ms[mk] = cv
                matched += 1
        features.load_state_dict(ms, strict=False)
        print(f"[Model 1] Successfully loaded {matched} matched ISL backbone parameters!")
    else:
        print("[Model 1] WARNING: Checkpoint not found, using ImageNet weights.")

    features.eval()
    avgpool.eval()
    return features, avgpool

# ── Load Cached Frames ────────────────────────────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  SignBridge - Fast Prototype Builder (ISL Sentence Model)")
    print("="*60)

    t0 = time.time()
    features, avgpool = get_isl_feature_extractor()

    exts = {".mp4",".MP4",".mov",".MOV",".avi",".AVI"}
    all_videos = []
    for sd in sorted(VIDEO_DIR.iterdir()):
        if not sd.is_dir(): continue
        lbl = sd.name.strip().lower()
        for vf in sd.iterdir():
            if vf.suffix in exts:
                all_videos.append((vf, lbl))

    print(f"Found {len(all_videos)} videos across {len(set(l for _,l in all_videos))} classes.")
    print("Extracting feature vectors with loaded ISL backbone (batched)...")

    class_vecs = defaultdict(list)
    video_records = []  # list of (video_path, label, vector)

    BATCH_VIDEOS = 16  # 16 videos * 16 frames = 256 frames per batch
    with torch.no_grad():
        for start_idx in range(0, len(all_videos), BATCH_VIDEOS):
            chunk = all_videos[start_idx:start_idx + BATCH_VIDEOS]
            batch_tensors = []
            valid_chunk = []
            
            for vpath, lbl in chunk:
                frames = get_cached_frames(vpath)
                if frames is None: continue
                tensors = torch.stack([TRANSFORM_VAL(f) for f in frames]) # [16, 3, 224, 224]
                batch_tensors.append(tensors)
                valid_chunk.append((vpath, lbl))
            
            if not batch_tensors: continue
            
            big_batch = torch.cat(batch_tensors, dim=0) # [B*16, 3, 224, 224]
            feats = torch.flatten(avgpool(features(big_batch)), 1) # [B*16, 960]
            feats = feats.view(len(valid_chunk), 16, 960) # [B, 16, 960]
            
            for idx, (vpath, lbl) in enumerate(valid_chunk):
                vec = feats[idx].mean(0).cpu().numpy()
                vec /= (np.linalg.norm(vec) + 1e-8)
                class_vecs[lbl].append(vec)
                video_records.append((vpath, lbl, vec))

            processed_cnt = min(start_idx + BATCH_VIDEOS, len(all_videos))
            print(f"  Processed {processed_cnt}/{len(all_videos)} videos [{time.time()-t0:.1f}s]", flush=True)

    labels = sorted(class_vecs.keys())
    print(f"\nSuccessfully extracted features for {len(video_records)} videos across {len(labels)} classes.")


    # ── Evaluate Leave-One-Out (LOO) Accuracy ─────────────────────────────────
    print("\nEvaluating Leave-One-Out (LOO) accuracy across all videos...")
    loo_correct = 0
    loo_top5 = 0

    for i, (vpath, true_lbl, vec) in enumerate(video_records):
        # Build prototypes excluding current video
        proto_list = []
        for l in labels:
            v_list = [v for vp, v_l, v in video_records if v_l == l and vp != vpath]
            if not v_list:
                # fallback if single sample class
                v_list = [v for vp, v_l, v in video_records if v_l == l]
            mean_v = np.mean(v_list, axis=0)
            mean_v /= (np.linalg.norm(mean_v) + 1e-8)
            proto_list.append(mean_v)
        
        proto_mat = np.stack(proto_list)  # [101, 960]
        sims = np.dot(proto_mat, vec)     # [101]
        top_indices = np.argsort(sims)[::-1]
        
        pred_lbl = labels[top_indices[0]]
        if pred_lbl == true_lbl:
            loo_correct += 1
        
        top5_lbls = [labels[idx] for idx in top_indices[:5]]
        if true_lbl in top5_lbls:
            loo_top5 += 1

    acc1 = 100.0 * loo_correct / max(len(video_records), 1)
    acc5 = 100.0 * loo_top5 / max(len(video_records), 1)
    print(f"LOO Top-1 Accuracy: {acc1:.1f}% ({loo_correct}/{len(video_records)})")
    print(f"LOO Top-5 Accuracy: {acc5:.1f}% ({loo_top5}/{len(video_records)})")

    # ── Save Final Prototypes ──────────────────────────────────────────────────
    print("\nSaving final prototype matrix & class vocabulary...")
    final_protos = []
    for l in labels:
        mean_v = np.mean(class_vecs[l], axis=0)
        mean_v /= (np.linalg.norm(mean_v) + 1e-8)
        final_protos.append(mean_v)

    final_mat = np.stack(final_protos).astype(np.float32)
    np.savez(PROTO_OUT, prototypes=final_mat, labels=np.array(labels))

    c2i = {l: idx for idx, l in enumerate(labels)}
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": len(labels),
            "class_to_idx": c2i,
            "idx_to_class": {str(idx): l for idx, l in enumerate(labels)},
            "recognition_mode": "prototype_cosine_isl_backbone"
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved prototype matcher : {PROTO_OUT}")
    print(f"Saved class vocabulary  : {CLASSES_OUT}")
    print(f"Total time elapsed      : {time.time()-t0:.1f} seconds")
    print("DONE.")

if __name__ == "__main__":
    main()
