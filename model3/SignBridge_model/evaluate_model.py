
import pandas as pd
import torch
import evaluate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "./model3_final"
TEST_FILE = "./test.csv"

# Load test data
df = pd.read_csv(TEST_FILE)

required_columns = {"sign_sequence", "english_sentence"}
if not required_columns.issubset(df.columns):
    raise ValueError("CSV must contain sign_sequence and english_sentence.")

df = df.dropna(subset=["sign_sequence", "english_sentence"])

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()




def translate(sequence):
    prefix = "translate Sign to English: "
    inputs = tokenizer(
        prefix + str(sequence),
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

# Generate predictions
references = df["english_sentence"].astype(str).tolist()
predictions = [
    translate(sequence)
    for sequence in df["sign_sequence"]
]

# Load evaluation metrics
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

bleu = bleu_metric.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

rouge = rouge_metric.compute(
    predictions=predictions,
    references=references
)

wer = wer_metric.compute(
    predictions=predictions,
    references=references
)

cer = cer_metric.compute(
    predictions=predictions,
    references=references
)

# Exact sentence match (strict case-insensitive comparison)
exact_matches = sum(
    pred.strip().casefold() == ref.strip().casefold()
    for pred, ref in zip(predictions, references)
)

sentence_accuracy = exact_matches / len(references) if references else 0.0

print("\n========== SIGNBRIDGE MODEL 3 EVALUATION ==========")

for i, (source, ref, pred) in enumerate(
    zip(df["sign_sequence"], references, predictions), start=1
):
    print(f"\nExample {i}")
    print("Sign sequence :", source)
    print("Reference     :", ref)
    print("Prediction    :", pred)
    print(
        "Exact match   :",
        pred.strip().casefold() == ref.strip().casefold()
    )

print("\n========== METRICS ==========")
print(f"BLEU                 : {bleu['score']:.2f} / 100")
print(f"ROUGE-1              : {rouge['rouge1']:.4f}")
print(f"ROUGE-2              : {rouge['rouge2']:.4f}")
print(f"ROUGE-L              : {rouge['rougeL']:.4f}")
print(f"Word Error Rate (WER): {wer:.4f}")
print(f"Character Error Rate : {cer:.4f}")
print(f"Exact sentence match : {sentence_accuracy * 100:.2f}%")
print(f"Test examples        : {len(references)}")
