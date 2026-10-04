import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ctc_lstm import CTCLSTM
from training.real_ctc_dataset import RealCTCDataset
from training.real_ctc_collate import collate_real_ctc


VAL_DIR = "data/continuous/new_val"
CLASSES_PATH = r"C:\Users\nihav\Downloads\final_model1_features\configs\classes_final.json"
CHECKPOINT_PATH = "models/checkpoints/new_lstm_ctc.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

class_to_idx = classes["class_to_idx"]
idx_to_class = {int(k): v for k, v in classes["idx_to_class"].items()}

dataset = RealCTCDataset(VAL_DIR, class_to_idx)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_real_ctc,
)

model = CTCLSTM(
    input_dim=960,
    hidden_dim=256,
    num_layers=2,
    num_classes=4764,
    dropout=0.3,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint)
model.eval()

blank_idx = 4764

total_frames = 0
blank_frames = 0
nonblank_frames = 0

predicted_class_counts = {}

with torch.no_grad():
    for features, feature_lengths, targets, target_lengths in loader:

        features = features.to(DEVICE)

        logits = model(features)

        predictions = torch.argmax(logits, dim=-1)

        for i in range(predictions.size(0)):
            length = feature_lengths[i].item()

            sequence = predictions[i, :length].cpu().tolist()

            total_frames += len(sequence)

            for idx in sequence:
                if idx == blank_idx:
                    blank_frames += 1
                else:
                    nonblank_frames += 1
                    predicted_class_counts[idx] = (
                        predicted_class_counts.get(idx, 0) + 1
                    )

print("=" * 50)
print("LSTM CTC Prediction Diagnostic")
print("=" * 50)

print(f"Total frames: {total_frames}")
print(f"Blank frames: {blank_frames}")
print(f"Non-blank frames: {nonblank_frames}")

print(f"Blank percentage: {blank_frames / total_frames * 100:.2f}%")
print(f"Non-blank percentage: {nonblank_frames / total_frames * 100:.2f}%")

print()
print(f"Unique predicted classes: {len(predicted_class_counts)}")
print()

print("Most frequent predicted classes:")

top_predictions = sorted(
    predicted_class_counts.items(),
    key=lambda x: x[1],
    reverse=True,
)[:20]

for class_id, count in top_predictions:
    print(
        f"{idx_to_class[class_id]}: "
        f"{count} frames"
    )

print("=" * 50)