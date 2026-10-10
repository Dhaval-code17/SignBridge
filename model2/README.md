# Model 2: Temporal Recognition & Sentence Prototype Matcher

## Overview
Model 2 receives the $960$-dimensional visual feature sequence from Model 1 and maps it into a $512$-dimensional temporal context embedding using a Multi-Head Self-Attention head (`SentenceClassifierHead`). The embedding is matched against pre-computed sentence prototypes for ISL CSLRT corpus recognition.

## Key Specifications
- **Input**: Visual feature sequence `[B, T, 960]`
- **Context Space**: $512$-dimensional L2-normalized attention vector
- **Vocabulary**: 101 full ISL sentence classes
- **Checkpoints**:
  - `models/checkpoints/sentence_classifier_head.pth`
  - `models/checkpoints/prototype_matcher.npz`

## File Map
- `classifier_head.py`: `SentenceClassifierHead` PyTorch module (temporal attention + linear projection).
- `prototype_builder.py`: L2 normalization and cosine similarity matching routines.
- `temporal_transformer.py`: Sequence Transformer encoder module.
