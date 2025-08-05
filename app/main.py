# API FastAPI pour la prédiction de sentiment
from fastapi import FastAPI
from app.predict import predict_sentiment

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API Sentiment Analysis OK"}

@app.post("/predict")
def predict(tweet: str):
    return {"sentiment": predict_sentiment(tweet)}
