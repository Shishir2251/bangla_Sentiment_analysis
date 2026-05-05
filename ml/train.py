import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from dataset_loader import load_data
from preprocess import clean_text


# =========================
# 1. LOAD DATA
# =========================
train_df, dev_df, test_df = load_data()

# Your dataset uses 'class_label'
train_df = train_df.rename(columns={"class_label": "label"})
dev_df   = dev_df.rename(columns={"class_label": "label"})
test_df  = test_df.rename(columns={"class_label": "label"})


# =========================
# 2. REMOVE NEUTRAL (BINARY)
# =========================
train_df = train_df[train_df['label'] != 'Neutral']
dev_df   = dev_df[dev_df['label'] != 'Neutral']
test_df  = test_df[test_df['label'] != 'Neutral']


# =========================
# 3. CLEAN TEXT
# =========================
train_df['text'] = train_df['text'].apply(clean_text)
dev_df['text']   = dev_df['text'].apply(clean_text)
test_df['text']  = test_df['text'].apply(clean_text)


# =========================
# 4. LABEL ENCODING
# =========================
le = LabelEncoder()

train_df['label'] = le.fit_transform(train_df['label'])
dev_df['label']   = le.transform(dev_df['label'])
test_df['label']  = le.transform(test_df['label'])

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


# =========================
# 5. MODEL + TOKENIZER
# =========================
MODEL_NAME = "csebuetnlp/banglabert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# =========================
# 6. TOKENIZATION
# =========================
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# =========================
# 7. CONVERT TO HF DATASET
# =========================
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
dev_dataset   = Dataset.from_pandas(dev_df[['text', 'label']])
test_dataset  = Dataset.from_pandas(test_df[['text', 'label']])

train_dataset = train_dataset.map(tokenize_batch, batched=True)
dev_dataset   = dev_dataset.map(tokenize_batch, batched=True)
test_dataset  = test_dataset.map(tokenize_batch, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# =========================
# 8. METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# =========================
# 9. TRAINING CONFIG
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    fp16=torch.cuda.is_available()
)


# =========================
# 10. TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# =========================
# 11. TRAIN
# =========================
trainer.train()


# =========================
# 12. EVALUATE
# =========================
results = trainer.evaluate(test_dataset)
print("Test Results:", results)


# =========================
# 13. SAVE MODEL
# =========================
SAVE_PATH = "../saved_model"

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ Model saved to {SAVE_PATH}")