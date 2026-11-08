from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / ".artifacts" / "ticket_classifier.joblib"
app = FastAPI(title="Support Ticket Triage")
model = None


class TicketRequest(BaseModel):
    text: str = Field(min_length=5, max_length=5000)


def load_model():
    global model
    if model is None and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    return model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": load_model() is not None}


@app.post("/triage")
def triage(ticket: TicketRequest):
    classifier = load_model()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained. Run `python train_model.py` first.",
        )

    probabilities = classifier.predict_proba([ticket.text])[0]
    best_index = probabilities.argmax()
    return {
        "category": str(classifier.classes_[best_index]),
        "confidence": float(probabilities[best_index]),
    }
