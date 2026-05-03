import re

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\s+", "", text)
    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text