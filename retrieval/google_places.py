"""
Retrieval for Engine A (general / Google Places).

Implements the pipeline from Section 8 of the spec:

    intent -> Text Search (New) with location bias + field mask
           -> small candidate set (20-100 places)

Grounded in the current official docs (checked 2026-09-26):
  - Endpoint:      https://places.googleapis.com/v1/places:searchText  (POST)
    https://developers.google.com/maps/documentation/places/web-service/text-search
  - Field mask is MANDATORY via the `X-Goog-FieldMask` header — omitting it
    is an error, and Google explicitly warns against requesting `*` in
    production because of cost and payload size.
    https://developers.google.com/maps/documentation/places/web-service/data-fields
  - Fields are billed by TIER (Essentials / Pro / Enterprise / Enterprise +
    Atmosphere) and you are billed at the HIGHEST tier touched by your field
    mask — see DESIGN.md "Google Places data policy" for the exact tier
    breakdown and the fields this module requests.
    https://developers.google.com/maps/documentation/places/web-service/usage-and-billing
  - CACHING: only `id` (place_id) may be cached/stored indefinitely.
    Everything else returned by this module (name, address, rating, hours,
    etc.) must be treated as short-lived — see the in-memory TTL cache below
    and DESIGN.md for what MUST NOT be written into a permanent database.
    https://developers.google.com/maps/documentation/places/web-service/policies

Do NOT expand the field mask (e.g. adding `reviews` or `photos`, which are
"Enterprise + Atmosphere") without re-checking current pricing — that tier is
materially more expensive and usually unnecessary for ranking.
"""
from __future__ import annotations

import logging
import time

import httpx

from config import settings
from models.schemas import Intent, NormalizedPlace
from normalization.normalized_place import from_google_place

logger = logging.getLogger("travelsathi.retrieval.google")

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Field mask: Essentials + Pro + Enterprise (NOT Enterprise+Atmosphere).
# Deliberately excludes `reviews`, `photos`, `editorialSummary` extras that
# push billing into the most expensive tier — add them only if a real product
# need justifies the cost.
FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.types",
    "places.primaryType",
    "places.businessStatus",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.currentOpeningHours",
    "places.googleMapsUri",
    "nextPageToken",
])

# Coarse category -> Places `includedType` mapping. Only used when exactly
# one strong category is present, to keep the text query natural-language
# driven (Google's ranking already does a good job combining text + type).
CATEGORY_TO_TYPE = {
    "cafe": "cafe",
    "food": "restaurant",
    "street_food": "restaurant",
    "shopping": "shopping_mall",
    "handicrafts": "store",
    "heritage": "tourist_attraction",
    "culture": "museum",
    "religious": "place_of_worship",
    "nightlife": "bar",
    "nature": "park",
    "entertainment": "tourist_attraction",
}

# In-memory result cache. Per Google's policy, only place_id may be stored
# indefinitely — this cache holds full (non-id) content, so it is TTL-bound
# and process-local, never persisted to disk/DB. See config.GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS.
_query_cache: dict[str, tuple[float, list[dict]]] = {}


def _build_text_query(intent: Intent) -> str:
    parts = []
    if intent.categories:
        parts.append(" ".join(intent.categories).replace("_", " "))
    else:
        parts.append("places to visit")
    if intent.location_text:
        parts.append(f"in {intent.location_text}")
    return " ".join(parts)


def _cache_key(intent: Intent) -> str:
    return f"{_build_text_query(intent)}|{intent.latitude}|{intent.longitude}|{intent.radius_km}"


def _mock_candidates(intent: Intent) -> list[dict]:
    """
    Deterministic, clearly-labeled synthetic candidates used when
    GOOGLE_PLACES_API_KEY is not configured, so the full pipeline (retrieval
    -> embeddings -> ranking) is runnable end-to-end without any API keys.
    Names are prefixed "[MOCK]" so nobody mistakes this for real data during
    a demo.
    """
    base_lat = intent.latitude or 12.9716  # Bangalore, if nothing else known
    base_lon = intent.longitude or 77.5946
    cats = intent.categories or ["cafe"]
    mocks = []
    for i, cat in enumerate(cats[:5] or ["cafe"]):
        for j in range(3):
            mocks.append({
                "id": f"MOCK_PLACE_{cat}_{j}",
                "displayName": {"text": f"[MOCK] {cat.title()} Spot {j+1} near {intent.location_text or 'you'}"},
                "formattedAddress": f"{intent.location_text or 'Unknown area'}, mock address {j+1}",
                "location": {"latitude": base_lat + 0.002 * (i + j), "longitude": base_lon + 0.002 * (i - j)},
                "types": [CATEGORY_TO_TYPE.get(cat, "point_of_interest"), cat],
                "primaryType": CATEGORY_TO_TYPE.get(cat, "point_of_interest"),
                "businessStatus": "OPERATIONAL",
                "rating": None,
                "userRatingCount": None,
                "priceLevel": None,
                "currentOpeningHours": {},
                "googleMapsUri": None,
            })
    return mocks


def _call_text_search(intent: Intent, page_token: str | None = None) -> dict:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.GOOGLE_PLACES_API_KEY or "",
        "X-Goog-FieldMask": FIELD_MASK,
    }
    body: dict = {
        "textQuery": _build_text_query(intent),
        "pageSize": 20,
    }
    if intent.latitude is not None and intent.longitude is not None:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": intent.latitude, "longitude": intent.longitude},
                "radius": (intent.radius_km or settings.GOOGLE_PLACES_DEFAULT_RADIUS_M / 1000) * 1000,
            }
        }
    if len(intent.categories) == 1 and intent.categories[0] in CATEGORY_TO_TYPE:
        body["includedType"] = CATEGORY_TO_TYPE[intent.categories[0]]
    if page_token:
        body["pageToken"] = page_token

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(TEXT_SEARCH_URL, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


def retrieve_google_candidates(intent: Intent, limit: int | None = None) -> list[NormalizedPlace]:
    """
    Stage-1 candidate retrieval for the Google engine. Paginates Text Search
    until `limit` (default settings.GOOGLE_PLACES_CANDIDATE_LIMIT, i.e. the
    "20-100 candidates" target from Section 8) candidates are collected, or
    Google stops returning a nextPageToken.
    """
    limit = limit or settings.GOOGLE_PLACES_CANDIDATE_LIMIT
    cache_key = _cache_key(intent)
    cached = _query_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < settings.GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS:
        logger.info("Google Places cache hit for query=%r", cache_key)
        raw_places = cached[1]
    else:
        raw_places: list[dict] = []
        if not settings.GOOGLE_PLACES_API_KEY:
            logger.warning("GOOGLE_PLACES_API_KEY not set — using mock candidates for demo purposes.")
            raw_places = _mock_candidates(intent)
        else:
            page_token = None
            try:
                while len(raw_places) < limit:
                    data = _call_text_search(intent, page_token)
                    raw_places.extend(data.get("places", []))
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
            except httpx.HTTPError as exc:
                logger.error("Google Places Text Search failed (%s); falling back to mock candidates.", exc)
                raw_places = _mock_candidates(intent)
        _query_cache[cache_key] = (time.time(), raw_places)

    normalized = [from_google_place(p) for p in raw_places[:limit]]
    logger.info("Google candidate retrieval: %d candidates for query=%r", len(normalized), cache_key)
    return normalized
