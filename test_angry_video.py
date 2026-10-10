"""
test_angry_video.py
===================
Tests sentence recognition on the sample video:
C:/Users/nihav/Downloads/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level/why are you angry/angry.MP4
"""

from pathlib import Path
import json, cv2, numpy as np, torch
from torchvision import models, transforms

TEST_VIDEO = Path(r"C:\Users\nihav\Downloads\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level\why are you angry\angry.MP4")
ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "models" / "checkpoints"
MODEL1_CKPT = CKPT_DIR / "mobilenet_v3_full_vocab_contrastive_best.pth"
PROTO_OUT   = CKPT_DIR / "prototype_matcher.npz"
CLASSES_OUT = CKPT_DIR / "temporal_classifier_classes.json"

IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def main():
    print("="*60)
    print("  Testing Sentence Recognition on angry.MP4")
    print("="*60)

    if not TEST_VIDEO.exists():
        print(f"Error: Video file not found at {TEST_VIDEO}")
        return

    if not PROTO_OUT.exists():
        print(f"Error: Prototype file not found at {PROTO_OUT}")
        return

    # Load prototype matcher
    pdata = np.load(PROTO_OUT)
    prototypes = pdata["prototypes"] # [101, D]
    labels = pdata["labels"]

    # Load MobileNetV3 ISL backbone
    bb = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    features = bb.features
    avgpool = bb.avgpool

    if MODEL1_CKPT.exists():
        ckpt = torch.load(MODEL1_CKPT, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        ms = features.state_dict()
        matched = 0
        for ck, cv in state.items():
            mk = ck.replace("backbone.", "")
            if mk in ms and ms[mk].shape == cv.shape:
                ms[mk] = cv; matched += 1
        features.load_state_dict(ms, strict=False)
        print(f"[Model 1] Loaded {matched} matched ISL backbone parameters.")

    features.eval(); avgpool.eval()

    # Read video frames
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = set(np.linspace(0, total-1, min(16, total)).astype(int).tolist())
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if idx in indices:
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE))
            frames.append(TRANSFORM(f))
        idx += 1
    cap.release()

    if not frames:
        print("Error: Could not read video frames.")
        return

    # Extract features
    with torch.no_grad():
        batch = torch.stack(frames) # [16, 3, 224, 224]
        feats = torch.flatten(avgpool(features(batch)), 1) # [16, 960]
        
        # Check if prototype dimension matches 960 or 512
        if prototypes.shape[1] == 960:
            query_vec = feats.mean(0).cpu().numpy()
        else:
            # If 512, load classifier head context projection
            HEAD_CKPT = CKPT_DIR / "sentence_classifier_head.pth"
            if HEAD_CKPT.exists():
                from training.train_fast_sentence_classifier import SentenceClassifierHead
                head = SentenceClassifierHead(in_dim=960, hidden_dim=512, num_classes=len(labels))
                h_ckpt = torch.load(HEAD_CKPT)
                head.load_state_dict(h_ckpt["model_state_dict"]); head.eval()
                _, context = head(feats.unsqueeze(0))
                query_vec = context.squeeze(0).cpu().numpy()
            else:
                query_vec = feats.mean(0).cpu().numpy()

        query_vec /= (np.linalg.norm(query_vec) + 1e-8)

    # Cosine Similarity Matching
    scores = np.dot(prototypes, query_vec) # [101]
    top5_idx = np.argsort(scores)[::-1][:5]

    print("\n--- Prediction Results ---")
    for rank, i in enumerate(top5_idx, 1):
        lbl = labels[i]
        sim = float(scores[i])
        marker = " <<< CORRECT MATCH!" if "angry" in lbl else ""
        print(f"Top-{rank}: {lbl} (Similarity: {sim:.4f}){marker}")

if __name__ == "__main__":
    main()
