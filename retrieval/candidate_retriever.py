"""
Engine router: decides which engine handles a request, and fetches that
engine's candidates. This is the ONLY place that knows both engines exist —
app.py just calls `get_candidates(intent, mode)`.
"""
from __future__ import annotations

import logging
from typing import Literal

from models.schemas import Intent, NormalizedPlace
from retrieval.google_places import retrieve_google_candidates
from retrieval.mumbai_retriever import retrieve_mumbai_candidates

logger = logging.getLogger("travelsathi.retrieval.router")

EngineName = Literal["mumbai", "google"]


def resolve_engine(intent: Intent, requested_mode: str = "auto") -> EngineName:
    """
    "auto" routing: use the Mumbai engine when the query resolves to Mumbai
    (via intent.city, set by intent extraction's area-keyword matching),
    otherwise fall back to the general Google engine. An explicit
    mode="mumbai"/"google" from the request always wins.
    """
    if requested_mode in ("mumbai", "google"):
        return requested_mode  # type: ignore[return-value]
    if intent.city and intent.city.lower() == "mumbai":
        return "mumbai"
    return "google"


def get_candidates(intent: Intent, engine: EngineName) -> list[NormalizedPlace]:
    if engine == "mumbai":
        return retrieve_mumbai_candidates(intent)
    return retrieve_google_candidates(intent)
