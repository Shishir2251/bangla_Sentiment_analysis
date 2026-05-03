from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Bangla Sentiment API")

app.include_router(router)