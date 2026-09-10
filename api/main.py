from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_prep import get_pairs
from src.intents import IntentClassifier
from src.retrieval import ReplyRetriever
from src.agent import SupportAgent

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "eval" / "results.csv"
DATA_PATH = ROOT / "data" / "processed" / "brand_pairs.csv"

pairs = get_pairs()
clf = IntentClassifier().fit(pairs["customer_text"].tolist())
retr = ReplyRetriever(
    pairs["customer_text"].tolist(),
    pairs["brand_text"].tolist(),
    corpus_intents=[clf.predict_one(t).intent for t in pairs["customer_text"]],
)
agent = SupportAgent(clf, retr)

app = FastAPI(
    title="AmazonHelp Support Agent API",
    version="1.1.0",
    description=(
        "Professional support agent for AmazonHelp on the TWCS dataset. "
        "This API exposes intent classification, grounded reply drafting, escalation, "
        "and evaluation metrics for human review through Swagger UI."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)


class MessageRequest(BaseModel):
    customer_text: str = Field(
        ..., min_length=3, max_length=2000,
        description="Customer message to classify, ground, and route."
    )


class MessageResponse(BaseModel):
    intent: str
    confidence: float
    action: str
    escalation_reason: str
    draft_reply: str
    grounded: bool
    grounding_reason: str


@app.get("/")
def root() -> dict:
    return {
        "service": "AmazonHelp Support Agent",
        "status": "ok",
        "docs": "/docs",
        "metrics": "/metrics",
        "brand": "AmazonHelp",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "brand": "AmazonHelp",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/sample")
def sample() -> dict:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="No processed dataset found. Run the pipeline first.")
    df = pd.read_csv(DATA_PATH)
    rows = df.head(5).to_dict(orient="records")
    return {
        "source": "real_twcs",
        "count": len(rows),
        "examples": rows,
    }


@app.get("/metrics")
def metrics() -> dict:
    if not EVAL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Run: python -m src.cli all",
        )
    df = pd.read_csv(EVAL_PATH)
    metrics_dict = df.to_dict(orient="records")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_source": df["judge_source"].iloc[0] if "judge_source" in df.columns else None,
        "systems": metrics_dict,
    }


@app.post("/predict", response_model=MessageResponse)
def predict(req: MessageRequest) -> MessageResponse:
    out = agent.handle(req.customer_text).to_dict()
    return MessageResponse(
        intent=out["intent"],
        confidence=out["intent_confidence"],
        action=out["action"],
        escalation_reason=out["escalation_reason"],
        draft_reply=out["draft_reply"],
        grounded=out["reply_grounded"],
        grounding_reason=out["grounding_reason"],
    )
