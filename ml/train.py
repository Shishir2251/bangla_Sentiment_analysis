import pandas as pd
import torch

from dataset_loader import load_data
from preprocess import clean_text

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# 🔹 Load dataset
train_df, dev_df, test_df = load_data()

# 🔹 Clean text
train_df['text'] = train_df['text'].apply(clean_text)
dev_df['text']   = dev_df['text'].apply(clean_text)
test_df['text']  = test_df['text'].apply(clean_text)

# 🔹 Encode labels
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['class_label'])
dev_df['label']   = le.transform(dev_df['class_label'])
test_df['label']  = le.transform(test_df['class_label'])

# 🔹 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

train_encodings = tokenize(train_df['text'])
dev_encodings   = tokenize(dev_df['text'])
test_encodings  = tokenize(test_df['text'])

# 🔹 Dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_df['label'].values)
dev_dataset   = SentimentDataset(dev_encodings, dev_df['label'].values)
test_dataset  = SentimentDataset(test_encodings, test_df['label'].values)

# 🔹 Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=3
)

# 🔹 Training config
training_args = TrainingArguments(
    output_dir="../results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../logs",
    load_best_model_at_end=True
)

# 🔹 Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# 🔹 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 🔹 Train
trainer.train()

# 🔹 Evaluate
results = trainer.evaluate(test_dataset)
print("Test Results:", results)

# 🔹 Save model
model.save_pretrained("../saved_model")
tokenizer.save_pretrained("../saved_model")