from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your fine-tuned model and tokenizer
MODEL_PATH = "./model3_final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def translate_sign_sequence(sequence: str) -> str:
    # Prefix must match what was used during training
    input_text = "translate Sign to English: " + sequence
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
    
    # Generate translation using beam search
    outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test cases from your sign vocabulary
test_sequences = [
    "ME COLLEGE GO TOMORROW",
    "YOU WHERE LIVE",
    "THIS COST HOW MUCH",
    "DOCTOR NEED ME"
]

print("--- TESTING MODEL 3 INFERENCE ---")
for seq in test_sequences:
    translation = translate_sign_sequence(seq)
    print(f"INPUT SIGN  : {seq}")
    print(f"TRANSLATION : {translation}\n")