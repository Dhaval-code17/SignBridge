
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "./model3_final"

print("Loading T5 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")


def translate_sign_sequence(sequence):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


sequence = "ME COLLEGE GO TOMORROW"

translation = translate_sign_sequence(sequence)

print("\nSign sequence:", sequence)
print("English translation:", translation)