from app.youtube import get_comments
from app.inference import predict

def analyze_video(video_id):
    comments = get_comments(video_id)

    results = []
    for c in comments:
        sentiment = predict(c)
        results.append({
            "text": c,
            "label": sentiment["label"],
            "confidence": sentiment["confidence"]
        })

    return results