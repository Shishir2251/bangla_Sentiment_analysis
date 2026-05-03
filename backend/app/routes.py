from fastapi import APIRouter
from app.schemas import SentimentRequest
from app.inference import predict

router =APIRouter()

@router.post("/predict")
def get_sentiment(req: SentimentRequest):
    return predict(req.text)