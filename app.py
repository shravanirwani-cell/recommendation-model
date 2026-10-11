"""
FastAPI interface for the TravelSATHI recommendation service (Section 21).

Run with:  uvicorn app:app --reload --port 8000
Then see README.md "HOW MY TEAMMATES INTEGRATE THIS" for exact request/
response shapes and sample curl commands.

Endpoints (only what's needed — Section 21):
  POST /recommend         -> mode="auto"|"mumbai"|"google", the one endpoint
                              the backend really needs day-to-day.
  POST /recommend/mumbai  -> force the Mumbai engine regardless of query text.
  POST /recommend/google  -> force the Google Places engine.
  POST /feedback          -> log an interaction event (Section 22).
  POST /embed/place       -> (debug/ops) embed one arbitrary text; used to
                              sanity-check the embedding backend and to help
                              pre-warm the model. Not on the hot request path.
  GET  /health            -> liveness/readiness probe for the team's backend
                              or an uptime check.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from embeddings.embedder import get_embedder
from feedback.feedback_store import record_feedback
from models.schemas import FeedbackEvent, Intent, RecommendRequest, RecommendResponse
from service import run_recommendation

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("travelsathi.app")

app = FastAPI(
    title="TravelSATHI Recommendation Service",
    version="1.0.0",
    description="Standalone retrieval + ranking microservice for TravelSATHI. See README.md.",
)

# Wide-open CORS for hackathon convenience — the team's backend is the only
# intended caller in production, so tighten `allow_origins` before a public
# deployment if this is ever exposed directly.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "embedding_backend": get_embedder().backend_name}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        return run_recommendation(request)
    except Exception as exc:
        logger.exception("recommend() failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc


@app.post("/recommend/mumbai", response_model=RecommendResponse)
def recommend_mumbai(request: RecommendRequest):
    forced = request.model_copy(update={"mode": "mumbai"})
    return recommend(forced)


@app.post("/recommend/google", response_model=RecommendResponse)
def recommend_google(request: RecommendRequest):
    forced = request.model_copy(update={"mode": "google"})
    return recommend(forced)


@app.post("/feedback")
def feedback(event: FeedbackEvent):
    record_feedback(event)
    return {"status": "recorded"}


@app.post("/embed/place")
def embed_place(text: str):
    """Debug/ops utility — embeds arbitrary text and returns the vector.
    Not used by the recommendation flow itself (embeddings there are
    computed per-request, in-memory, inside ranker.rank_candidates)."""
    embedder = get_embedder()
    vector = embedder.encode([text])[0]
    return {"backend": embedder.backend_name, "dimensions": len(vector), "vector": vector.tolist()}
