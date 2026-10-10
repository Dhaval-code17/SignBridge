# Model 3: Sign-to-English Natural Language Translation

## Overview
Model 3 uses a fine-tuned sequence-to-sequence T5 Transformer (`t5-small`) to convert ISL sign sequences/glosses into natural, grammatically correct English sentences.

## Key Specifications
- **Architecture**: `T5ForConditionalGeneration` (`t5-small`)
- **Input Format**: `"translate Sign to English: <GLOSS_SEQUENCE>"`
- **Output Format**: Grammatical English sentence (e.g., `"I will go to college tomorrow."`)
- **Checkpoint Location**: `model3/SignBridge_model/model3_final/` (`model.safetensors` + tokenizer)

## File Map
- `SignBridge_model/model3_final/`: Pre-trained weights, tokenizer, and config files.
- `SignBridge_model/infer.py`: Sequence translation inference function.
- `SignBridge_model/train.py`: T5 fine-tuning script.
