import numpy as np, random
from pathlib import Path
from collections import defaultdict

ROOT       = Path(".")
VIDEO_DIR  = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
FEAT_CACHE = ROOT / "data" / "model2_sentence_features"
exts       = {".mp4",".MP4",".mov",".MOV",".avi",".AVI"}

label_to_seqs = defaultdict(list)
for sd in sorted(VIDEO_DIR.iterdir()):
    if not sd.is_dir(): continue
    label = sd.name.strip().lower()
    for vf in sd.iterdir():
        if vf.suffix not in exts: continue
        safe  = vf.stem[:40] + "_" + sd.name[:20].replace(" ","_")
        cache = FEAT_CACHE / (safe + ".npy")
        if not cache.exists(): continue
        label_to_seqs[label].append(np.load(cache))   # keep full [T, 960]

classes = sorted(label_to_seqs.keys())
print(f"Classes: {len(classes)}  Total videos: {sum(len(v) for v in label_to_seqs.values())}")

# Mean-pooled vectors
label_to_vecs = {}
for lbl, seqs in label_to_seqs.items():
    vecs = []
    for seq in seqs:
        v = seq.mean(0); v /= (np.linalg.norm(v)+1e-8)
        vecs.append(v)
    label_to_vecs[lbl] = vecs

# Intra vs inter cosine similarity
intra, inter = [], []
for lbl in classes:
    vecs = label_to_vecs[lbl]
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            intra.append(float(vecs[i] @ vecs[j]))
all_vecs = [(lbl, v) for lbl in classes for v in label_to_vecs[lbl]]
random.seed(0)
for _ in range(2000):
    (l1,v1),(l2,v2) = random.sample(all_vecs,2)
    if l1 != l2: inter.append(float(v1@v2))

print(f"\n--- Feature Separability (Mean-Pooled) ---")
print(f"Intra-class sim : mean={np.mean(intra):.4f}  std={np.std(intra):.4f}")
print(f"Inter-class sim : mean={np.mean(inter):.4f}  std={np.std(inter):.4f}")
print(f"Gap (intra-inter): {np.mean(intra)-np.mean(inter):.4f}")

# Check feature norms
norms = [np.linalg.norm(seq.mean(0)) for seqs in label_to_seqs.values() for seq in seqs]
print(f"\nRaw feature vector norms: mean={np.mean(norms):.2f} std={np.std(norms):.2f}")

# Why are you angry class
lbl = "why are you angry"
seqs = label_to_seqs.get(lbl, [])
print(f"\nClass '{lbl}': {len(seqs)} videos")
if len(seqs)>=2:
    vecs = label_to_vecs[lbl]
    mat  = np.stack(vecs)
    sims = mat @ mat.T
    print(f"  Pairwise cosine sims: {np.round(sims[np.triu_indices(len(vecs),1)],3).tolist()}")
    print(f"  Frame counts: {[s.shape[0] for s in seqs]}")
