# SignBridge — Model 1
## Indian Sign Language Visual Recognition & Feature Extraction

SignBridge is an AI-powered system designed to convert **Indian Sign Language (ISL)** gestures into meaningful English sentences and speech.

This repository section contains the contribution of **Person 1**, responsible for the **visual sign recognition and feature extraction stage (Model 1)**.

Model 1 processes video/camera input, detects the visual information present in sign-language frames, and converts the frames into a sequence of compact numerical feature vectors.

These feature vectors are passed to **Model 2**, which performs temporal sign-sequence recognition.

---

# 1. Person 1 Contribution

Person 1 is responsible for the complete **visual processing and feature extraction pipeline**.

### Responsibilities

- Video preprocessing
- Frame sampling
- Image preprocessing
- Data augmentation
- CISLR dataset preparation
- Class mapping
- MobileNetV3-Large visual encoder
- Model 1 training
- Contrastive representation learning
- Visual feature extraction
- Continuous video feature extraction
- Model 1 → Model 2 interface
- Model 1 evaluation
- Model 1 inference
- Camera/video integration
- Model 1 testing
- Feature validation
- Performance and latency testing

The output of Model 1 is a temporal sequence of visual feature vectors.

### Frozen Model 1 → Model 2 Interface

```text
[T, 960] float32
```

Where:

- `T` = number of sampled video frames
- `960` = visual feature dimension
- `float32` = feature data type

---

# 2. Complete SignBridge Architecture

```text
                    ┌──────────────────────┐
                    │   Camera / Video     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Preprocessing      │
                    │                      │
                    │ • Frame Sampling     │
                    │ • Resize 224×224     │
                    │ • Normalization      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MODEL 1         │
                    │                      │
                    │ MobileNetV3-Large    │
                    │ Visual Encoder       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Visual Features      │
                    │                      │
                    │ [T, 960] float32     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MODEL 2         │
                    │                      │
                    │ Temporal Transformer │
                    │ + CTC                │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sign Sequence        │
                    │                      │
                    │ ["I", "GOOD",        │
                    │  "TODAY"]            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MODEL 3         │
                    │                      │
                    │ T5 / mT5             │
                    │ English Translation  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Natural English      │
                    │ Sentence             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text-to-Speech       │
                    └──────────────────────┘
```

---

# 3. Model 1 Role

Model 1 is the **visual encoder** of SignBridge.

Its job is **not** to generate the final English sentence.

Instead, Model 1 converts visual information from sign-language videos into numerical representations that preserve useful visual information.

The pipeline is:

```text
Input Video
     ↓
Frames
     ↓
Preprocessing
     ↓
MobileNetV3-Large
     ↓
960-dimensional visual feature
     ↓
Sequence of features
     ↓
[T, 960]
     ↓
Model 2
```

Model 1 therefore acts as the bridge between:

```text
Visual Video
      ↓
Numerical Representation
      ↓
Temporal Sign Recognition
```

---

# 4. Model 1 Input

Model 1 accepts video frames.

Input can come from:

- Live camera
- Uploaded video
- Dataset video

Each frame is converted to:

```text
RGB
224 × 224
ImageNet normalized
```

For continuous feature extraction, the default sampling rate is approximately:

```text
8 FPS
```

The number of frames is not fixed during continuous feature extraction.

Therefore:

```text
Video A → [T1, 960]
Video B → [T2, 960]
Video C → [T3, 960]
```

where:

```text
T1 ≠ T2 ≠ T3
```

may occur depending on video duration.

---

# 5. Input Preprocessing

Each frame passes through the following pipeline:

```text
Original Video
      ↓
Frame Decoding
      ↓
Frame Sampling
      ↓
RGB Conversion
      ↓
Resize to 224 × 224
      ↓
ImageNet Normalization
      ↓
MobileNetV3-Large
```

### Input Resolution

```text
224 × 224
```

### Color Format

```text
RGB
```

### Normalization

ImageNet normalization is used so that the input is compatible with the pretrained MobileNetV3-Large backbone.

---

# 6. Frame Sampling

The system uses sequential video decoding with uniform temporal sampling.

For continuous feature extraction, the default sampling rate is:

```text
8 FPS
```

This produces an ordered sequence of frames.

Maintaining temporal order is important because Model 2 uses the sequence of Model 1 features to understand the temporal structure of a sign.

For example:

```text
Frame 1
   ↓
Feature 1

Frame 2
   ↓
Feature 2

Frame 3
   ↓
Feature 3

...

Frame T
   ↓
Feature T
```

The resulting feature sequence is:

```text
[
    feature_1,
    feature_2,
    feature_3,
    ...
    feature_T
]
```

with shape:

```text
[T, 960]
```

---

# 7. Data Augmentation

Training augmentation was used to improve robustness.

The training preprocessing pipeline includes:

- Resize
- Small random rotation
- Brightness/contrast variation
- Gaussian blur

### Horizontal Flip

Horizontal flipping is intentionally **disabled**.

This is important because hand orientation and movement direction can contain semantic information in sign language.

The evaluation pipeline uses deterministic preprocessing without training augmentation.

---

# 8. Dataset

Model 1 was developed using the **CISLR dataset**.

The dataset contains Indian Sign Language video samples and their corresponding gloss labels.

### Dataset Statistics

```text
Total videos:             7,050
Unique glosses:           4,764
Supervised videos:        7,049
Unlabeled/malformed row:  1

Training videos:          5,686
Validation videos:        1,363
```

The final vocabulary contains:

```text
4,764 gloss classes
```

The raw dataset is intentionally **not included in this Git repository** because of its size and dataset distribution requirements.

---

# 9. Final Vocabulary

The final Model 1 vocabulary contains:

```text
4,764 glosses
```

The class mapping is stored in:

```text
configs/classes_final.json
```

This file is critical for Model 1 → Model 2 compatibility.

It should remain tracked in Git.

The class mapping must not be independently regenerated by different team members because Model 2 must use the same vocabulary/index mapping.

---

# 10. Dataset Split

The final development split contains:

```text
Training:
5,686 videos

Validation:
1,363 videos
```

The training set covers all:

```text
4,764 glosses
```

The validation set contains:

```text
1,363 videos
1,363 represented glosses
```

This is a highly challenging vocabulary because many glosses have very few examples.

The dataset contains many classes with only one or a small number of samples.

Therefore, conventional large-scale classification performance should be interpreted carefully.

---

# 11. Model 1 Architecture

Model 1 uses:

```text
MobileNetV3-Large
```

as the visual backbone.

The backbone is initialized using ImageNet-pretrained weights.

The architecture can be summarized as:

```text
Input Frame
224 × 224 × 3
      ↓
MobileNetV3-Large
      ↓
Global Average Pooling
      ↓
960-dimensional feature
      ↓
Visual Representation
```

The important output from the backbone is:

```text
960 dimensions
```

---

# 12. Why MobileNetV3-Large?

MobileNetV3-Large was selected because SignBridge is intended to support practical camera-based inference.

Important considerations include:

- Lightweight architecture
- Good mobile/edge suitability
- Fast inference
- Lower computational requirements than many larger CNNs
- Strong pretrained ImageNet representation
- Suitable feature-extraction backbone
- Easy integration with PyTorch
- Suitable for continuous video processing

The architecture provides a practical balance between:

```text
Accuracy
    +
Inference Speed
    +
Model Size
    +
Deployment Feasibility
```

---

# 13. Representation Learning

The final full-vocabulary experiment used a supervised contrastive representation-learning approach.

The training architecture contains:

```text
MobileNetV3-Large
        ↓
960-D feature
        ↓
128-D projection head
        ↓
Contrastive training
```

The projection head was used during training.

However, the projection head is **not part of the frozen Model 1 → Model 2 interface**.

The production handoff uses:

```text
960-D backbone feature
```

and not:

```text
128-D projection
```

and not classifier logits.

---

# 14. Important Model 1 Output Contract

The Model 1 → Model 2 interface is frozen as:

```text
[T, 960] float32
```

This is the most important integration rule in the repository.

### Example

For a video producing 24 sampled frames:

```text
[24, 960]
```

For a video producing 36 sampled frames:

```text
[36, 960]
```

For a video producing 16 sampled frames:

```text
[16, 960]
```

The first dimension represents temporal order.

The second dimension represents the visual feature vector.

---

# 15. What Model 2 Must Consume

Person 2 / Model 2 should consume:

```text
[T, 960] float32
```

Example:

```python
features.shape
# (T, 960)
```

The sequence must remain ordered:

```text
feature[0]
feature[1]
feature[2]
...
feature[T-1]
```

Model 2 should **not** expect:

```text
128-D projection
```

or:

```text
classifier logits
```

or:

```text
class probabilities
```

or:

```text
one-hot class labels
```

The interface is:

```text
Model 1
   ↓
[T, 960] float32
   ↓
Model 2
```

---

# 16. Continuous Feature Extraction

The main production feature-extraction implementation is:

```text
training/final_model1_handoff.py
```

It loads the trained contrastive checkpoint, extracts the MobileNetV3-Large backbone representation, and generates temporal feature sequences.

The output structure is:

```text
data/processed/<output>/

├── features/
│   ├── seq_00000.npy
│   ├── seq_00001.npy
│   ├── seq_00002.npy
│   └── ...
│
├── metadata.json
├── labels.json
└── sequences.csv
```

---

# 17. Feature File Format

Each `.npy` feature file contains:

```text
[T, 960]
```

with datatype:

```text
float32
```

Example:

```python
import numpy as np

features = np.load("seq_00000.npy")

print(features.shape)
print(features.dtype)
```

Expected:

```text
(T, 960)
float32
```

---

# 18. Feature Validation

The final Model 1 handoff was validated for:

- Correct number of sequences
- Correct feature dimension
- Correct dtype
- Finite numerical values
- NaN detection
- Inf detection
- Vocabulary coverage
- Label consistency
- Video/feature sequence mapping

### Training Feature Library

```text
Sequences:             5,686
Vocabulary:            4,764 classes
Classes represented:   4,764
Total extracted frames: 217,903
Feature shape:         [T,960]
Dtype:                 float32
NaN/Inf:               PASS
```

### Validation Feature Library

```text
Sequences:             1,363
Classes represented:   1,363
Feature shape:         [T,960]
Dtype:                 float32
NaN/Inf:               PASS
```

---

# 19. Model Checkpoint

The trained Model 1 checkpoint is:

```text
models/checkpoints/mobilenet_v3_full_vocab_contrastive_best.pth
```

The checkpoint contains the trained MobileNetV3-based representation-learning model.

The checkpoint is intentionally **not stored in Git** because model weights are large.

Therefore:

```text
GitHub
   ↓
Source code + configuration
```

while:

```text
Separate artifact storage
   ↓
Model checkpoint
```

The checkpoint must be shared separately with the team when Model 2 / Model 3 integration is performed.

---

# 20. Why the Model Checkpoint Is Not in Git

The repository intentionally excludes:

```text
*.pth
*.pt
*.ckpt
*.onnx
```

This keeps the Git repository lightweight and avoids storing large binary model artifacts directly in the source repository.

The source code remains reproducible as long as the corresponding checkpoint is supplied separately.

---

# 21. Model 1 Evaluation

A visual embedding retrieval benchmark was performed on:

```text
200 validation videos
```

using the final full-vocabulary contrastive encoder.

The benchmark used:

```text
4,764 gloss vocabulary
```

and training examples as visual references.

### Results

```text
Top-1 Accuracy: 21.00%

Top-5 Accuracy: 29.00%
```

These results represent a:

```text
Visual Embedding Retrieval Benchmark
```

and **not** the final end-to-end SignBridge translation accuracy.

Therefore, these numbers should not be interpreted as:

```text
"SignBridge has 21% accuracy."
```

The correct interpretation is:

> Model 1 visual embedding retrieval achieved 21% Top-1 and 29% Top-5 accuracy on a 200-video validation benchmark across 4,764 glosses.

---

# 22. Important Evaluation Limitation

The CISLR dataset has a highly imbalanced class distribution.

There are:

- Many singleton classes
- Several low-frequency classes
- A smaller number of repeated classes

This makes conventional 4,764-class softmax classification difficult.

The final Model 1 therefore focuses on learning a reusable visual representation rather than treating the visual encoder as the complete SignBridge system.

Model 2 is responsible for learning temporal sign sequences from these visual representations.

Model 3 is responsible for natural-language translation.

---

# 23. Model 1 Does Not Perform Final Translation

Model 1 does **not** generate:

```text
English sentences
```

It does **not** perform:

```text
T5 translation
```

It does **not** perform:

```text
Text-to-Speech
```

Its responsibility ends at:

```text
Video
 ↓
Visual Features
 ↓
[T,960]
```

The rest of the pipeline is handled by the other team members.

---

# 24. Project Structure

The relevant repository structure is:

```text
SignBridge/
│
├── configs/
│   ├── classes_final.json
│   └── feature_extractor_config.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
│
├── inference/
│
├── models/
│   ├── checkpoints/
│   ├── evaluation/
│   ├── mobilenet_model.py
│   ├── cnn_baseline.py
│   └── ...
│
├── preprocessing/
│   ├── create_class_mapping.py
│   ├── create_manifest.py
│   ├── create_splits.py
│   ├── create_full_cislr_manifest.py
│   ├── create_final_dev_split.py
│   ├── transforms.py
│   ├── video_sampling.py
│   └── ...
│
├── training/
│   ├── dataset.py
│   ├── feature_extractor.py
│   ├── train_mobilenet.py
│   ├── train_final_mobilenet.py
│   ├── train_final_contrastive.py
│   ├── final_model1_handoff.py
│   ├── extract_continuous_features.py
│   ├── evaluate_contrastive_encoder.py
│   └── ...
│
├── ui/
│
├── signbridge_server.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

Large generated files such as datasets, feature caches, model weights, and local environments are excluded from Git.

---

# 25. Important Files

## Model Architecture

```text
models/mobilenet_model.py
```

Contains the MobileNetV3-based Model 1 architecture.

## Dataset

```text
training/dataset.py
```

Handles loading and preparing video samples for training.

## Frame Sampling

```text
preprocessing/video_sampling.py
```

Handles sequential video decoding and frame sampling.

## Image Transformations

```text
preprocessing/transforms.py
```

Contains training and evaluation preprocessing.

## Final Model Training

```text
training/train_final_contrastive.py
```

Used for the final full-vocabulary contrastive representation-learning experiment.

## Final Model 1 Handoff

```text
training/final_model1_handoff.py
```

This is the most important script for Model 1 → Model 2 integration.

It:

1. Loads the trained checkpoint
2. Extracts the MobileNetV3 backbone
3. Removes the projection head from the handoff
4. Generates 960-dimensional features
5. Preserves temporal order
6. Saves `[T,960]` float32 sequences
7. Generates metadata
8. Generates label information
9. Validates generated features

---

# 26. Configuration Files

## Class Mapping

```text
configs/classes_final.json
```

Contains the final:

```text
4,764-class vocabulary
```

This file must be shared with Model 2.

## Feature Contract

```text
configs/feature_extractor_config.json
```

Contains the Model 1 feature-extraction configuration.

Important parameters include:

```text
Feature dimension: 960
Output format:    [T,960]
Datatype:         float32
Sampling rate:    8 FPS
Input resolution: 224 × 224
Color format:     RGB
Normalization:    ImageNet
```

---

# 27. Installing the Environment

On Windows:

```powershell
cd D:\signbridge_model1
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected GPU support:

```text
CUDA: True
```

if the system has a correctly configured NVIDIA CUDA environment.

---

# 28. Running Model 1 Feature Extraction

The final handoff script supports feature extraction.

Example:

```powershell
python -m training.final_model1_handoff extract `
  --csv data/processed/final_dev_train.csv `
  --output data/processed/final_model1_features_train `
  --checkpoint models/checkpoints/mobilenet_v3_full_vocab_contrastive_best.pth `
  --classes configs/classes_final.json
```

For validation:

```powershell
python -m training.final_model1_handoff extract `
  --csv data/processed/final_dev_val.csv `
  --output data/processed/final_model1_features_val `
  --checkpoint models/checkpoints/mobilenet_v3_full_vocab_contrastive_best.pth `
  --classes configs/classes_final.json
```

Check the exact available command-line options with:

```powershell
python -m training.final_model1_handoff --help
```

---

# 29. Verifying Model 1 Features

After extraction, run the verification command provided by the handoff script.

Example:

```powershell
python -m training.final_model1_handoff verify `
  --features data/processed/final_model1_features_train `
  --classes configs/classes_final.json
```

For validation:

```powershell
python -m training.final_model1_handoff verify `
  --features data/processed/final_model1_features_val `
  --classes configs/classes_final.json
```

The verification process checks:

```text
Feature dimensions
Data type
Finite values
NaN values
Inf values
Sequence count
Class coverage
Label validity
```

---

# 30. Loading a Feature Sequence

Example Python code:

```python
import numpy as np

features = np.load(
    "data/processed/final_model1_features_train/features/seq_00000.npy"
)

print("Shape:", features.shape)
print("Dtype:", features.dtype)
print("Finite:", np.isfinite(features).all())
```

Expected output:

```text
Shape: (T, 960)
Dtype: float32
Finite: True
```

---

# 31. Model 2 Integration

Person 2 should connect Model 1 to the temporal model as follows:

```text
Video
  ↓
Model 1
  ↓
[T,960]
  ↓
Temporal Transformer
  ↓
CTC
  ↓
Sign Sequence
```

Conceptually:

```python
features = model1(video)

# features:
# [T, 960]

sign_sequence = model2(features)
```

The exact temporal model implementation belongs to Person 2.

Model 1 should not contain assumptions about the internal architecture of Model 2.

---

# 32. Integration Contract

The following interface should remain unchanged:

```text
Input:
Video / Camera Frames

Output:
[T,960] float32
```

The following should **not** be changed without team agreement:

```text
Feature dimension = 960
```

```text
Feature dtype = float32
```

```text
Temporal dimension = T
```

```text
Temporal order = preserved
```

```text
Classifier logits = not part of interface
```

```text
128-D projection = not part of interface
```

Therefore:

```text
                  MODEL 1
                     │
                     ▼
              ┌──────────────┐
              │  [T, 960]    │
              │   float32    │
              └──────┬───────┘
                     │
                     ▼
                  MODEL 2
```

---

# 33. Live Camera Inference

The project also contains a local SignBridge server:

```text
signbridge_server.py
```

Run:

```powershell
python signbridge_server.py
```

The local interface is available at:

```text
http://localhost:8001
```

The UI supports Model 1 camera/video processing.

The current interface can display:

```text
Model 1 feature shape
Sampling rate
Latency
Feature extraction status
Recognition/reference information
```

---

# 34. Current Model 1 UI Pipeline

The current UI pipeline is approximately:

```text
Camera
   ↓
Browser
   ↓
SignBridge Server
   ↓
Frame preprocessing
   ↓
MobileNetV3-Large
   ↓
960-D feature
   ↓
Model 1 output
```

The UI can confirm that Model 1 has generated the visual representation.

Complete sentence translation requires:

```text
Model 2
+
Model 3
+
TTS
```

to be integrated.

Therefore, the current Model 1 UI should not be described as the complete SignBridge translation system.

---

# 35. Local Server Features

The current Model 1 server supports:

- Real Model 1 loading
- Camera inference
- Uploaded video processing
- MobileNetV3-Large backbone
- 960-dimensional visual features
- Continuous feature extraction
- Model 1 recognition/reference functionality
- Feature shape reporting
- Latency measurement
- UI integration

The server loads the Model 1 backbone without using:

```text
Projection head
```

or:

```text
Classifier head
```

for the Model 2 feature interface.

---

# 36. Reference Library

The current Model 1 recognition/demo setup uses the extracted training feature library.

The library contains:

```text
5,686 training videos
```

and covers:

```text
4,764 glosses
```

The feature library is intentionally excluded from Git because of its size.

It can be regenerated using the Model 1 feature-extraction pipeline.

---

# 37. Why the Feature Library Is Not in Git

Generated feature files can become very large.

The repository therefore ignores:

```text
data/processed/
features/
models/features/
```

This keeps the source repository manageable.

The source code and configuration needed to regenerate the features are retained.

---

# 38. Model 1 Development Experiments

During development, multiple Model 1 approaches were evaluated.

These included:

```text
CNN baseline
      ↓
MobileNetV3-Large Stage 1
      ↓
MobileNetV3-Large Stage 2
      ↓
Full 4,764-class classification
      ↓
Full-vocabulary contrastive representation learning
```

The final Model 1 handoff uses the full-vocabulary contrastive representation-learning checkpoint.

---

# 39. CNN Baseline

A small CNN baseline was implemented to establish a basic visual classification reference.

The CNN contains multiple convolutional layers followed by:

```text
Batch Normalization
ReLU
Pooling
Adaptive Average Pooling
Dropout
Classifier
```

The CNN baseline was primarily used for development comparison.

It is **not** the final Model 1 architecture.

---

# 40. MobileNetV3 Development

A MobileNetV3-Large development model was also trained in stages.

The development process included:

```text
Stage 1:
Frozen backbone + train classifier

Stage 2:
Unfreeze final backbone blocks
+
Fine-tune classifier
```

These experiments were useful for evaluating the architecture on the prototype vocabulary.

However, those development metrics should not be treated as the final full-vocabulary Model 1 performance.

---

# 41. Final Full-Vocabulary Approach

The final vocabulary contains:

```text
4,764 glosses
```

Because many classes have very few examples, a conventional 4,764-way softmax classification approach performed poorly.

A representation-learning approach was therefore explored.

The final contrastive encoder was trained to learn useful visual representations.

The resulting 960-dimensional backbone features are used for the Model 1 → Model 2 interface.

---

# 42. Important Metric Disclaimer

Some earlier prototype experiments produced higher classification numbers on a smaller vocabulary.

Those experiments are **not** the final clean full-vocabulary benchmark.

In particular, some 39-class development experiments had overlap between development data and official test data.

Therefore, they should **not** be reported as the final generalization performance of SignBridge.

The final documented Model 1 benchmark is:

```text
Vocabulary:          4,764 glosses
Validation benchmark: 200 videos

Top-1:               21%
Top-5:               29%
```

and should be described specifically as:

```text
Visual embedding retrieval performance
```

---

# 43. Reproducibility

The Model 1 pipeline is designed to be reproducible using:

```text
Source code
      +
Configuration
      +
CISLR dataset
      +
Model checkpoint
```

The following are version-controlled:

```text
Python source code
Configuration files
Preprocessing code
Training code
Evaluation code
Server/UI code
README
Requirements
```

Large binary artifacts are excluded:

```text
Dataset
Model weights
Feature caches
ONNX files
Virtual environment
Generated archives
```

---

# 44. Files That Must Be Shared With Model 2

For Model 2 integration, Person 2 needs:

### Required

```text
configs/classes_final.json
```

```text
configs/feature_extractor_config.json
```

```text
Model 1 checkpoint
```

```text
training/final_model1_handoff.py
```

### Optional

Pre-extracted feature library:

```text
final_model1_features_train
final_model1_features_val
```

The feature library is useful for immediate experimentation but can also be regenerated from the source dataset and checkpoint.

---

# 45. What Person 3 Needs

Person 3 is responsible for the language-generation portion of SignBridge.

Person 3 does not need to modify Model 1 internals.

The integration point is:

```text
Model 1
    ↓
[T,960] float32
    ↓
Model 2
    ↓
Sign sequence
    ↓
Model 3
    ↓
English sentence
    ↓
TTS
```

Person 3 can therefore integrate the complete pipeline without changing the Model 1 feature contract.

---

# 46. Recommended Team Repository Structure

The shared SignBridge repository can be organized as:

```text
SignBridge/
│
├── person-1-model1
├── person-2-model2
├── person-3-model3
└── main
```

Suggested workflow:

```text
Person 1
    ↓
person-1-model1
    ↓
Review
    ↓
main

Person 2
    ↓
person-2-model2
    ↓
Review
    ↓
main

Person 3
    ↓
person-3-model3
    ↓
Review
    ↓
main
```

The final integrated system should be tested after merging all three components.

---

# 47. GitHub Branch

The recommended branch for Person 1 is:

```text
person-1-model1
```

Person 1's contribution should be pushed to this branch rather than directly modifying `main`.

This allows Person 2 and Person 3 to integrate the Model 1 contribution safely.

---

# 48. Git Ignore Policy

The repository intentionally excludes:

```text
.venv/
__pycache__/

*.pth
*.pt
*.ckpt
*.onnx

data/raw/
data/processed/
features/
models/features/

*.zip
logs/
```

This prevents large datasets, model weights, generated features, and local environments from entering Git history.

The following file **must remain tracked**:

```text
configs/classes_final.json
```

because it defines the final Model 1 → Model 2 vocabulary mapping.

---

# 49. Security and Repository Hygiene

Do not commit:

```text
.env
API keys
Passwords
Access tokens
Private credentials
Personal datasets
Large model artifacts
Local virtual environments
```

The repository should contain only the source/configuration necessary to reproduce and integrate Model 1.

---

# 50. Model 1 Limitations

The current Model 1 has several limitations.

### 1. Large Vocabulary

The system handles:

```text
4,764 glosses
```

with a highly imbalanced dataset.

Many classes have very few training examples.

### 2. Visual Representation Is Not Complete Translation

Model 1 only produces visual features.

It does not understand the complete temporal sentence structure by itself.

That is the responsibility of Model 2.

### 3. Final English Generation Is Outside Model 1

Natural-language generation is handled by Model 3.

### 4. Current Retrieval Benchmark

The current benchmark reports:

```text
21% Top-1
29% Top-5
```

on a 200-video visual embedding retrieval benchmark.

This is **not** an end-to-end translation score.

### 5. Feature Artifacts Are External

Large feature libraries and checkpoints are not stored directly in Git.

They must be shared separately or regenerated.

---

# 51. Future Improvements

Potential future improvements to Model 1 include:

- More CISLR training samples
- Better handling of singleton classes
- Improved data augmentation
- Stronger contrastive learning
- Hard-negative mining
- Larger batch sizes
- Memory-bank based contrastive learning
- Better temporal sampling
- Multi-frame augmentation
- Hand/pose-aware features
- Optical-flow information
- Transformer-based visual encoders
- Knowledge distillation
- Model quantization
- ONNX/TensorRT optimization
- Mobile deployment
- Edge inference optimization
- Better robustness to lighting
- Better robustness to camera angle
- Better robustness to background changes

These are future enhancements and are not required for the current Model 1 handoff.

---

# 52. Testing Checklist

Before integrating Model 1 with Model 2, verify:

```text
[✓] Model checkpoint loads
[✓] MobileNetV3-Large backbone loads
[✓] Input resolution = 224 × 224
[✓] RGB preprocessing works
[✓] ImageNet normalization works
[✓] Feature dimension = 960
[✓] Feature dtype = float32
[✓] Temporal order preserved
[✓] No NaN values
[✓] No Inf values
[✓] Class mapping available
[✓] Training features generated
[✓] Validation features generated
[✓] Model 2 receives [T,960]
```

---

# 53. Model 1 → Model 2 Final Contract

This is the most important section for integration.

## Input to Model 1

```text
Video / Camera Frames
```

## Model 1 preprocessing

```text
RGB
224 × 224
ImageNet normalization
~8 FPS continuous sampling
```

## Model 1 backbone

```text
MobileNetV3-Large
```

## Model 1 visual representation

```text
960-dimensional
```

## Model 1 output

```text
[T,960]
```

## Output datatype

```text
float32
```

## Temporal ordering

```text
Preserved
```

## Projection head

```text
NOT part of Model 2 interface
```

## Classifier logits

```text
NOT part of Model 2 interface
```

Therefore:

```text
                  MODEL 1
                     │
                     ▼
              ┌──────────────┐
              │  [T, 960]    │
              │   float32    │
              └──────┬───────┘
                     │
                     ▼
                  MODEL 2
```

---

# 54. Quick Start

Clone the repository:

```powershell
git clone https://github.com/Dhaval-code17/SignBridge.git
```

Enter the repository:

```powershell
cd SignBridge
```

Checkout the Model 1 branch:

```powershell
git checkout person-1-model1
```

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import torch, torchvision; print('Torch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

# 55. Run Model 1 Server

After supplying the Model 1 checkpoint and required feature/reference artifacts:

```powershell
python signbridge_server.py
```

Open:

```text
http://localhost:8001
```

The UI can then be used for Model 1 camera/video testing.

---

# 56. Example End-to-End Integration

The complete SignBridge pipeline will eventually look like:

```text
                 CAMERA / VIDEO
                       │
                       ▼
              ┌─────────────────┐
              │   PREPROCESSING │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     MODEL 1     │
              │ MobileNetV3-L   │
              └────────┬────────┘
                       │
                       ▼
                  [T, 960]
                   float32
                       │
                       ▼
              ┌─────────────────┐
              │     MODEL 2     │
              │ Transformer+CTC │
              └────────┬────────┘
                       │
                       ▼
                 SIGN SEQUENCE
                       │
                       ▼
              ┌─────────────────┐
              │     MODEL 3     │
              │     T5/mT5      │
              └────────┬────────┘
                       │
                       ▼
                ENGLISH SENTENCE
                       │
                       ▼
              ┌─────────────────┐
              │      TTS        │
              └────────┬────────┘
                       │
                       ▼
                     SPEECH
```

---

# 57. Responsibility Boundaries

## Person 1 — Model 1

Responsible for:

```text
Video
 ↓
Frame preprocessing
 ↓
MobileNetV3-Large
 ↓
960-D visual features
```

## Person 2 — Model 2

Responsible for:

```text
[T,960]
 ↓
Temporal modeling
 ↓
CTC
 ↓
Sign sequence
```

## Person 3 — Model 3

Responsible for:

```text
Sign sequence
 ↓
T5/mT5
 ↓
Natural English
 ↓
TTS
```

---

# 58. Final Status of Person 1

The Model 1 pipeline has been implemented and validated for the current project handoff.

### Dataset

```text
CISLR
7,050 total videos
4,764 glosses
```

### Final Development Data

```text
5,686 training videos
1,363 validation videos
```

### Model

```text
MobileNetV3-Large
```

### Representation

```text
960-dimensional
```

### Model 2 Interface

```text
[T,960] float32
```

### Sampling

```text
~8 FPS
```

### Final Visual Retrieval Benchmark

```text
Top-1: 21%
Top-5: 29%
```

### Benchmark Size

```text
200 validation videos
```

### Feature Validation

```text
NaN/Inf:            PASS
Feature dimension:  PASS
Dtype:              PASS
Vocabulary coverage: PASS
```

---

# 59. Conclusion

Person 1's Model 1 provides the **visual representation layer** of SignBridge.

The system takes Indian Sign Language video input and converts it into a temporally ordered sequence of 960-dimensional visual feature vectors.

The final interface is:

```text
Video
   ↓
MobileNetV3-Large
   ↓
[T,960] float32
   ↓
Model 2
```

This separation allows the visual encoder, temporal sign recognizer, and language-generation model to be developed independently while maintaining a fixed interface between the components.

The most important integration requirement is:

```text
MODEL 1 OUTPUT = [T,960] float32
```

Model 2 should consume this sequence directly while preserving temporal order.

Model 1 is therefore the **visual foundation** of the SignBridge pipeline, while Model 2 and Model 3 complete temporal sign recognition and natural-language translation.

---

# 60. Authors / Team

## SignBridge

AI-powered Indian Sign Language to English translation system.

### Person 1 — Model 1

Responsibilities:

- Visual sign recognition
- Video preprocessing
- MobileNetV3-Large
- Feature extraction
- Contrastive representation learning
- Model 1 evaluation
- Model 1 → Model 2 integration interface
- Camera/video inference

### Person 2 — Model 2

Responsibilities:

- Temporal sign recognition
- Transformer
- CTC
- Sign sequence generation

### Person 3 — Model 3

Responsibilities:

- Natural-language translation
- T5/mT5
- English sentence generation
- Text-to-Speech integration

---

# License

This project is released under the **MIT License**.

See:

```text
LICENSE
```

for details.