from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI(title="AI Sentiment Analysis API")

# Lazy-loaded tokenizer and model
tokenizer = None
model = None

API_KEY = os.getenv("API_KEY")

class TextRequest(BaseModel):
    text: str


def load_model():
    """Load tokenizer and model into module-level variables."""
    global tokenizer, model
    try:
        # Switching to a model that supports Turkish sentiment
        model_name = "savasy/bert-base-turkish-sentiment-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        logging.info(f"Model {model_name} loaded successfully")
    except Exception:
        logging.exception("Failed to load model/tokenizer")
        tokenizer = None
        model = None


@app.on_event("startup")
def on_startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(
    request: TextRequest,
    x_api_key: str = Header(...)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    if tokenizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )

    with torch.no_grad():
        inputs = tokenizer(
            request.text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    # savasy/bert-base-turkish-sentiment-cased labels:
    # 0 -> negative, 1 -> neutral, 2 -> positive
    return {
        "negative": float(probs[0][0]),
        "neutral": float(probs[0][1]),
        "positive": float(probs[0][2])
    }