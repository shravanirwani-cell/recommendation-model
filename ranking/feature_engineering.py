"""
Per-candidate feature scores, each in [0, 1] or None (= unknown).

None is a first-class value here, not a bug: Section 18 requires that
missing data never be silently treated as zero or as "worst case". A place
with an unknown rating should not automatically outrank OR underrank a
place with a known mediocre rating — scoring.py decides how to combine
knowns/unknowns; this module only ever reports what it can actually
determine from the evidence in a NormalizedPlace.

Feature -> Section 15 classification (see DESIGN.md Step 7 for the full
writeup of why each one landed where it did):
  - distance_score, budget_score, time_score, group_score  -> SOFT
    constraints / ranking features (penalize, don't disqualify, because the
    underlying data is frequently incomplete or approximate).
  - quality_score, local_score                              -> ranking
    features (pure signal, no notion of "violating" them).
  - category/tag matching is intentionally NOT a separate feature here —
    it's already inside the text that gets embedded (embedding_text), so a
    separate category_score would double-count the same signal that
    semantic_score already captures.
  - Hard, deterministic constraints (permanently closed, provably closed
    today, group too large for capacity) live in scoring.py's
    `check_hard_constraints`, not here — they either pass or the candidate
    is dropped before scoring even runs.
"""
from __future__ import annotations

import math

from models.schemas import Intent, NormalizedPlace


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_km(intent: Intent, place: NormalizedPlace) -> float | None:
    if None in (intent.latitude, intent.longitude, place.latitude, place.longitude):
        return None
    return _haversine_km(intent.latitude, intent.longitude, place.latitude, place.longitude)


def distance_score(intent: Intent, place: NormalizedPlace) -> float | None:
    d = distance_km(intent, place)
    if d is None:
        return None
    scale = (intent.radius_km or 5.0) * 1.5  # score decays to ~0 by ~1.5x the requested radius
    return max(0.0, 1.0 - d / scale) if scale > 0 else None


def _effective_budget_per_person(intent: Intent) -> float | None:
    if not intent.budget or intent.budget.amount is None:
        return None
    if intent.budget.basis.value == "total" and intent.group_size:
        return intent.budget.amount / intent.group_size
    # "per_person" or "unknown" — treat as per-person (the more conservative/
    # common phrasing in the target queries, e.g. "₹2000 each").
    return intent.budget.amount


def budget_score(intent: Intent, place: NormalizedPlace) -> float | None:
    if place.cost.is_free:
        return 1.0
    budget_pp = _effective_budget_per_person(intent)
    place_cost = place.cost.max_inr if place.cost.max_inr is not None else place.cost.min_inr
    if budget_pp is None or place_cost is None:
        return None  # unknown on either side — no basis to score
    if place_cost <= budget_pp:
        return 1.0
    overage_ratio = (place_cost - budget_pp) / budget_pp
    return max(0.0, 1.0 - overage_ratio)  # linearly penalize going over budget


def time_score(intent: Intent, place: NormalizedPlace) -> float | None:
    """Compares the user's AVAILABLE time to the PROVIDER's estimated visit duration.
    Never conflate the two — see Section 19."""
    if intent.duration_minutes is None or place.duration_minutes is None:
        return None
    if place.duration_minutes <= intent.duration_minutes:
        return 1.0
    overage_ratio = (place.duration_minutes - intent.duration_minutes) / intent.duration_minutes
    return max(0.0, 1.0 - overage_ratio)


def group_score(intent: Intent, place: NormalizedPlace) -> float | None:
    if intent.group_size is None or place.capacity.max_people is None:
        return None
    if intent.group_size > place.capacity.max_people:
        return 0.0  # normally already excluded by the hard constraint check
    if place.capacity.min_people and intent.group_size < place.capacity.min_people:
        return 0.7  # group is smaller than the experience is really designed for — mild penalty
    return 1.0


def quality_score(place: NormalizedPlace) -> float | None:
    if place.rating is None:
        return None
    base = max(0.0, min(1.0, place.rating / 5.0))
    # Confidence shrinkage: a 5.0 rating from 2 reviews shouldn't outrank a
    # 4.5 from 500 reviews. Mild log-scale boost capped at +0.05.
    if place.review_count:
        confidence_bonus = min(0.05, math.log10(1 + place.review_count) / 100)
        base = min(1.0, base + confidence_bonus)
    return base


def local_score(place: NormalizedPlace) -> float | None:
    """Section 11: optional future signal. Returns None until the Local
    Knowledge Layer has real contributor data — never fabricated."""
    if place.local.local_score is None:
        return None
    return max(0.0, min(1.0, place.local.local_score))
