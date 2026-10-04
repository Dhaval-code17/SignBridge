import os
import json
import torch
import torch.nn as nn
from training.train_temporal_classifier import TemporalTransformerClassifier


class ContinuousRecognizer:
    def __init__(
        self,
        checkpoint_path=None,
        classes_path=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if checkpoint_path is None:
            candidate_paths = [
                os.path.join(root_dir, "models", "temporal_classifier_best.pth"),
                os.path.join(root_dir, "models", "checkpoints", "temporal_classifier_best.pth"),
                "models/temporal_classifier_best.pth",
                "models/checkpoints/temporal_classifier_best.pth",
            ]
            checkpoint_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

        if classes_path is None:
            candidate_paths = [
                os.path.join(root_dir, "models", "temporal_classifier_classes.json"),
                os.path.join(root_dir, "models", "checkpoints", "temporal_classifier_classes.json"),
                "models/temporal_classifier_classes.json",
                "models/checkpoints/temporal_classifier_classes.json",
            ]
            classes_path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])

        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                class_data = json.load(f)
            self.idx_to_class = {
                int(k): v for k, v in class_data["idx_to_class"].items()
            }
            num_classes = class_data["num_classes"]
        else:
            self.idx_to_class = {}
            num_classes = 4764

        self.model = TemporalTransformerClassifier(
            input_dim=960,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            num_classes=num_classes,
            dropout=0.2,
        )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, features):
        """
        features:
            numpy array or torch tensor of shape [T, 960]

        returns:
            list of recognized sign labels, e.g. ["ME COLLEGE GO TOMORROW"]
        """
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)

        if features.ndim == 2:
            features = features.unsqueeze(0)  # [1, T, 960]

        if features.shape[2] != 960:
            raise ValueError(f"Expected feature dim 960, got {features.shape[2]}")

        features = features.to(self.device)

        with torch.no_grad():
            logits = self.model(features)  # [1, num_classes]
            pred_idx = torch.argmax(logits, dim=-1).item()

        sign_label = self.idx_to_class.get(pred_idx, f"CLASS_{pred_idx}")
        return [sign_label]