"""
Retrieval for Engine B (Mumbai). Loads raw provider records from either a
local JSON file (zero-setup default) or the team's MySQL database, and
converts them to NormalizedPlace via normalization.normalized_place.

This module deliberately does the candidate generation (a coarse geo/keyword
filter) BEFORE normalization+embedding, per Section 8's "don't embed
everything for every query" principle — even though the seed dataset here is
small enough that it wouldn't matter yet, the same code path will matter once
the dataset grows to hundreds/thousands of providers.
"""
from __future__ import annotations

import json
import logging
import math
from functools import lru_cache

from config import settings
from models.schemas import Intent, NormalizedPlace
from normalization.normalized_place import from_mumbai_record

logger = logging.getLogger("travelsathi.retrieval.mumbai")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _load_json_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("places", [])


def _load_raw_records() -> list[dict]:
    if settings.MUMBAI_DATA_SOURCE == "mysql":
        from retrieval.mysql_retriever import fetch_mumbai_places_raw

        return fetch_mumbai_places_raw()
    return _load_json_dataset(settings.MUMBAI_DATA_PATH)


def retrieve_mumbai_candidates(intent: Intent, limit: int = 100) -> list[NormalizedPlace]:
    """
    Stage-1 candidate retrieval for the Mumbai engine (Section 7, Stage 1).

    Applies only cheap, purely geographic/keyword pre-filtering here — real
    constraint filtering (budget, group size, etc.) happens later in
    ranking/scoring.py against the full NormalizedPlace, once we've paid the
    (tiny, for this dataset size) cost of normalizing.
    """
    raw_records = _load_raw_records()
    places = [from_mumbai_record(r) for r in raw_records]

    if intent.latitude is not None and intent.longitude is not None:
        radius_km = intent.radius_km or 5.0
        in_range = []
        for p in places:
            if p.latitude is None or p.longitude is None:
                in_range.append(p)  # no coords -> don't drop it, just can't geo-filter it
                continue
            d = _haversine_km(intent.latitude, intent.longitude, p.latitude, p.longitude)
            if d <= radius_km * 2.5:  # generous — real ranking penalizes distance, doesn't hard-cut here
                in_range.append(p)
        if in_range:
            places = in_range

    if intent.categories:
        wanted = {c.lower() for c in intent.categories}

        def _matches_category(p: NormalizedPlace) -> bool:
            place_terms = {c.lower() for c in p.categories} | {t.lower() for t in p.tags}
            return bool(wanted & place_terms) or any(
                w in " ".join(place_terms) for w in wanted
            )

        category_matches = [p for p in places if _matches_category(p)]
        # If the category filter would eliminate everything, keep the wider
        # geo-filtered set instead — better to semantically rank a broader
        # set than to return zero results because of a keyword mismatch.
        if category_matches:
            places = category_matches

    logger.info("Mumbai candidate retrieval: %d candidates for intent categories=%s", len(places), intent.categories)
    return places[:limit]
