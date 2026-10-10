import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)

# 1. Load Dataset
df = pd.read_csv("train.csv")
raw_dataset = Dataset.from_pandas(df)

# Use small validation split for small datasets
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)

# 2. Load Pretrained T5 Model & Tokenizer
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 3. Preprocessing Function
PREFIX = "translate Sign to English: "

def preprocess_function(examples):
    inputs = [PREFIX + seq for seq in examples["sign_sequence"]]
    targets = examples["english_sentence"]
    
    model_inputs = tokenizer(inputs, max_length=64, truncation=True)
    labels = tokenizer(text_target=targets, max_length=64, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = split_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=["sign_sequence", "english_sentence"]
)

# 4. Evaluation Metrics Setup
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_labels_bleu = [[label] for label in decoded_labels]
    
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "bleu": round(bleu_result["score"], 2),
        "rouge1": round(rouge_result["rouge1"], 2)
    }

# 5. Optimized Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,          # Increased learning rate for faster weight adaptation
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=45,          # Increased epochs so the model learns the mappings thoroughly
    predict_with_generate=True,
    logging_steps=5,
    report_to="none"
)

# 6. Trainer Execution
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

print("🚀 Starting Fine-Tuning in WSL...")
trainer.train()

# 7. Save Model & Tokenizer Artifacts
model.save_pretrained("./model3_final")
tokenizer.save_pretrained("./model3_final")
print("✅ Model successfully trained and saved to './model3_final'!")