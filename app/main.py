from __future__ import annotations
import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import HTTPException

from pathlib import Path

from .schemas import PredictRequest, PredictResponse, FeedbackRequest
from .inference import predict_one

# --- FastAPI setup ---
app = FastAPI(title="Sentiment API (Keras)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Application Insights (simple SDK) ---
APPINSIGHTS_KEY = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY")
telemetry_client = None
if APPINSIGHTS_KEY:
    try:
        from applicationinsights import TelemetryClient
        telemetry_client = TelemetryClient(APPINSIGHTS_KEY)
    except Exception:
        telemetry_client = None

ROOT = Path(__file__).resolve().parents[0]  # adapte si besoin
INDEX = ROOT / "static" / "index.html"

@app.get("/", include_in_schema=False)
def home():
    if INDEX.exists():
        return FileResponse(INDEX)
    raise HTTPException(
        status_code=404,
        detail=f'UI non trouvée. Placez "{INDEX}" dans le repo.'
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    res = predict_one(req.text)

    if telemetry_client:
        telemetry_client.track_event(
            "prediction",
            {"sentiment": res["sentiment"], "model_version": res["model_version"]},
        )
        telemetry_client.flush()
    return res

@app.post("/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """
    L'utilisateur nous indique si la prédiction était correcte (correct=True) ou non.
    On trace dans Application Insights afin de créer des alertes sur les erreurs.
    """
    if telemetry_client:
        # On tronque le texte pour éviter d'envoyer trop de données
        txt = (req.text or "")[:500]
        note = (req.note or "")[:200]
        telemetry_client.track_event(
            "bad_prediction" if not req.correct else "feedback_ok",
            {
                "predicted": req.predicted,
                "correct": str(req.correct),
                "note": note,
                "text": txt,
            },
        )
        telemetry_client.flush()
    return {"status": "stored"}
