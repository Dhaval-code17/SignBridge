# Model 1: Visual Feature Extraction

## Overview
Model 1 converts video frames into compact 960-dimensional feature vectors using a fine-tuned MobileNetV3-Large CNN backbone.

## Key Specifications
- **Backbone Architecture**: MobileNetV3-Large (Feature Extractor + AvgPool)
- **Input**: Video frame batch `[B, 3, 224, 224]`
- **Output**: Visual feature tensor `[B, 960]`
- **Sampling Rate**: 8 FPS (24-32 frames per 3-4s video sequence)
- **Checkpoint Location**: `models/checkpoints/mobilenet_v3_full_vocab_contrastive_best.pth`

## File Map
- `encoder.py`: PyTorch `Model1Encoder` definition and `TRANSFORM` image preprocessing.
- `finetune.py`: Script for fine-tuning MobileNetV3 backbone on sentence-level videos.
