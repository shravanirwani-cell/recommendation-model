"""
Framework-agnostic core of the recommendation service.

`run_recommendation()` is the ENTIRE pipeline (Sections 7-8) in one call. It
has no dependency on FastAPI, so it can be:
  - called directly from app.py's HTTP handlers,
  - called directly from tests (no HTTP client, no server, needed),
  - reused from a CLI script or a notebook while iterating on ranking.

This separation is deliberate: HTTP concerns (status codes, request
parsing) stay in app.py; recommendation logic stays here and in
retrieval/ranking/embeddings.
"""
from __future__ import annotations

import logging

from intent.gemini_intent import extract_intent
from models.schemas import Intent, RecommendRequest, RecommendResponse
from retrieval.candidate_retriever import get_candidates, resolve_engine
from ranking.ranker import rank_candidates
from config import settings

logger = logging.getLogger("travelsathi.service")


def run_recommendation(request: RecommendRequest, intent_override: Intent | None = None) -> RecommendResponse:
    """
    intent_override lets /recommend/mumbai and /recommend/google (Section 21)
    force the engine while still running full intent extraction on the query
    text, and lets tests inject a fixed Intent without needing real geocoding.
    """
    intent = intent_override or extract_intent(request.query)

    # A request-level lat/lon (from the app's GPS, say) overrides whatever
    # intent extraction guessed from text, when provided.
    if request.latitude is not None and request.longitude is not None:
        intent = intent.model_copy(update={"latitude": request.latitude, "longitude": request.longitude})

    engine = resolve_engine(intent, request.mode)
    top_k = request.top_k or settings.DEFAULT_TOP_K
    top_k = min(top_k, settings.MAX_TOP_K)

    candidates = get_candidates(intent, engine)
    recommendations, engine_meta = rank_candidates(intent, candidates, engine, top_k=top_k)

    response = RecommendResponse(
        mode=engine,
        intent=intent,
        recommendations=recommendations,
        engine_meta=engine_meta,
    )
    logger.info(
        "query_id=%s engine=%s user_id=%s -> %d recommendations",
        response.query_id, engine, request.user_id, len(recommendations),
    )
    return response
