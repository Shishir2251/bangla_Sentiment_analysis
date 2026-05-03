import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH= r"C:\Users\Shishir\Projects\bangla-sentiment-classification\saved_model"

#Load tokenize and model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()