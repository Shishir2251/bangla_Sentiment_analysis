from fastapi import APIRouter
from app.schemas import SentimentRequest
from app.inference import predict
from app.services import analyze_video

router =APIRouter()

@router.post("/predict")
def get_sentiment(req: SentimentRequest):
    return predict(req.text)

@router.get("/youtube/{video_id}")
def youtube_sentiment(video_id: str):
    return analyze_video(video_id)