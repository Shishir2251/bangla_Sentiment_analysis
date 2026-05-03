import torch
from model.model_loader import model, tokenizer

# IMPORTANT: must match your LabelEncoder order
labels = ["negative", "neutral", "positive"]

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    return {
        "label": labels[pred],
        "confidence": float(probs[0][pred])
    }