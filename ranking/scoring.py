"""
Hard constraint checks + the weighted scoring function (Section 15).

check_hard_constraints() implements ONLY constraints we can check
deterministically and are confident enough about to justify dropping a
candidate entirely (Section 18: "Numerical constraints must be checked
deterministically", and never on an ambiguous/unknown value). Everything
else is a ranking feature applied in `score_candidate` — penalized, not
excluded, because provider data here is frequently incomplete.
"""
from __future__ import annotations

from datetime import datetime, timezone

from config import settings
from models.schemas import Intent, NormalizedPlace, ScoreBreakdown
from ranking import feature_engineering as fe


def check_hard_constraints(intent: Intent, place: NormalizedPlace, now: datetime | None = None) -> str | None:
    """Returns None if the place passes, or a human-readable reason it was excluded."""
    now = now or datetime.now(timezone.utc)

    if place.business_status in ("CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY"):
        return f"Excluded: business_status={place.business_status}"

    if place.opening_hours.weekly_closure:
        today_name = now.strftime("%A")
        if today_name in place.opening_hours.weekly_closure:
            return f"Excluded: closed on {today_name}"

    if intent.group_size is not None and place.capacity.max_people is not None:
        if intent.group_size > place.capacity.max_people:
            return f"Excluded: group of {intent.group_size} exceeds capacity of {place.capacity.max_people}"

    if intent.time_constraint == "open now" and place.opening_hours.open_now is False:
        return "Excluded: currently closed and user requires open-now"

    return None


def score_candidate(
    intent: Intent, place: NormalizedPlace, semantic_score: float
) -> tuple[ScoreBreakdown, float]:
    """Returns (score_breakdown, constraint_score)."""
    """
    Combines semantic_score (from embeddings/vector_store.py — computed
    outside this function since it needs the whole candidate batch at once)
    with the structured features into one weighted final_score.

    Unknown features are treated as neutral (0.5) rather than 0 or 1, so
    missing data pulls a candidate toward the middle of the pack instead of
    being punished or rewarded for something we simply don't know.
    """
    d_score = fe.distance_score(intent, place)
    b_score = fe.budget_score(intent, place)
    t_score = fe.time_score(intent, place)
    g_score = fe.group_score(intent, place)
    q_score = fe.quality_score(place)
    l_score = fe.local_score(place)

    NEUTRAL = 0.5
    weighted_sum = (
        settings.WEIGHT_SEMANTIC * semantic_score
        + settings.WEIGHT_DISTANCE * (d_score if d_score is not None else NEUTRAL)
        + settings.WEIGHT_BUDGET * (b_score if b_score is not None else NEUTRAL)
        + settings.WEIGHT_TIME * (t_score if t_score is not None else NEUTRAL)
        + settings.WEIGHT_GROUP * (g_score if g_score is not None else NEUTRAL)
        + settings.WEIGHT_QUALITY * (q_score if q_score is not None else NEUTRAL)
        + settings.WEIGHT_LOCAL * (l_score if l_score is not None else NEUTRAL)
    )

    non_semantic_weight = (
        settings.WEIGHT_DISTANCE + settings.WEIGHT_BUDGET + settings.WEIGHT_TIME
        + settings.WEIGHT_GROUP + settings.WEIGHT_QUALITY + settings.WEIGHT_LOCAL
    )
    constraint_composite = (weighted_sum - settings.WEIGHT_SEMANTIC * semantic_score)
    constraint_score = constraint_composite / non_semantic_weight if non_semantic_weight > 0 else NEUTRAL

    return ScoreBreakdown(
        semantic_score=round(semantic_score, 4),
        distance_score=round(d_score, 4) if d_score is not None else None,
        budget_score=round(b_score, 4) if b_score is not None else None,
        time_score=round(t_score, 4) if t_score is not None else None,
        group_score=round(g_score, 4) if g_score is not None else None,
        quality_score=round(q_score, 4) if q_score is not None else None,
        local_score=round(l_score, 4) if l_score is not None else None,
        final_score=round(min(1.0, max(0.0, weighted_sum)), 4),
    ), round(min(1.0, max(0.0, constraint_score)), 4)


def build_match_reasons(intent: Intent, place: NormalizedPlace, breakdown: ScoreBreakdown) -> list[str]:
    """Human-readable explanations (Section 20/23 — "why did this place get this score")."""
    reasons: list[str] = []

    overlap = set(c.lower() for c in intent.categories) & set(
        c.lower() for c in (place.categories + place.tags)
    )
    if overlap:
        reasons.append(f"Matches {', '.join(sorted(overlap))} preference")
    elif breakdown.semantic_score and breakdown.semantic_score > 0.5:
        reasons.append("Semantically related to your request")

    if intent.location_text:
        reasons.append(f"Located in/near {intent.location_text}")

    if intent.group_size and (breakdown.group_score or 0) >= 0.99:
        reasons.append(f"Suitable for {intent.group_size} people")

    if intent.budget and (breakdown.budget_score or 0) >= 0.99:
        reasons.append("Within your stated budget")
    elif place.cost.is_free:
        reasons.append("Free / no entry cost")

    if breakdown.distance_score is not None and breakdown.distance_score > 0.7:
        reasons.append("Close to your specified location")

    if place.rating and place.rating >= 4.0:
        reasons.append(f"Highly rated ({place.rating}/5)")

    if place.local.local_tips:
        reasons.append("Backed by local tips")

    return reasons or ["General match for your request"]
