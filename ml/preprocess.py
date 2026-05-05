import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)  # keep Bangla only
    return text.strip()