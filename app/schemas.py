from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw tweet text")

class PredictResponse(BaseModel):
    sentiment: str
    proba_neg: float
    proba_pos: float
    model_version: str

class FeedbackRequest(BaseModel):
    text: str
    predicted: str           # "pos" ou "neg" renvoyé par l'API
    correct: bool            # True si l'utilisateur confirme ; False si erreur
    note: Optional[str] = None
