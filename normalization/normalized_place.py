"""
Converts raw Google Places records and raw Mumbai dataset rows into the
single NormalizedPlace shape (models.schemas.NormalizedPlace).

This is the ONLY file that needs to know the field names Google Places
uses, and the only file that needs to know the Mumbai dataset's column
names. Everything downstream (embeddings, ranking) only ever sees
NormalizedPlace. If Google renames a field or you add a column to the
Mumbai dataset, you change exactly one function here.

Section 18 (missing data) is handled here, not in the ranker: parsing
happens once, at the boundary, so a malformed/ambiguous source value
becomes an explicit `None` rather than a silently-wrong number that
propagates through scoring.
"""
from __future__ import annotations

import re

from models.schemas import Capacity, Cost, LocalKnowledge, NormalizedPlace, OpeningHours


# --------------------------------------------------------------------------
# Shared parsing helpers (Section 18 — never guess, never silently coerce)
# --------------------------------------------------------------------------

def parse_capacity(raw: str | int | None) -> Capacity:
    """
    '1-4'  -> min=1, max=4   (max CAPACITY, not "typically 1")
    '4'    -> min=4, max=4
    None   -> everything None (unknown, not zero)
    """
    if raw is None:
        return Capacity(raw=None)
    if isinstance(raw, int):
        return Capacity(min_people=raw, max_people=raw, raw=str(raw))
    raw_str = str(raw).strip()
    m = re.match(r"^\s*(\d+)\s*[-–—]\s*(\d+)\s*$", raw_str)
    if m:
        return Capacity(min_people=int(m.group(1)), max_people=int(m.group(2)), raw=raw_str)
    m = re.match(r"^\s*(\d+)\s*\+?\s*$", raw_str)
    if m:
        n = int(m.group(1))
        return Capacity(min_people=n, max_people=n, raw=raw_str)
    return Capacity(raw=raw_str)  # unparseable free text — keep it, don't guess numbers


def parse_cost(
    amount: float | None = None,
    cost_basis: str | None = None,
    free_text: str | None = None,
) -> Cost:
    """
    Handles the documented trap: cost="Free entry; spend varies" does NOT
    mean the whole experience is ₹0 — it means entry is free but spend at
    the venue is open-ended. We only ever set is_free=True when the text
    unambiguously says so AND there is no "spend varies" / "extra" caveat.
    """
    if free_text:
        text = free_text.strip()
        lowered = text.lower()
        has_caveat = any(w in lowered for w in ["varies", "extra", "additional", "depending", "approx"])
        looks_free = "free" in lowered
        is_free = True if (looks_free and not has_caveat) else (None if looks_free else False)
        amt_match = re.search(r"(\d+(?:\.\d+)?)", text)
        parsed_amount = float(amt_match.group(1)) if amt_match else None
        return Cost(
            min_inr=None if is_free else parsed_amount,
            max_inr=None if is_free else parsed_amount,
            basis=cost_basis or "unknown",
            is_free=is_free,
            note=text,
        )
    if amount is None:
        return Cost(basis=cost_basis or "unknown")
    return Cost(min_inr=amount, max_inr=amount, basis=cost_basis or "unknown", is_free=(amount == 0))


def parse_weekly_closure(raw: str | list[str] | None) -> list[str] | None:
    """None/'' -> None (UNKNOWN). Never default to '[]' meaning 'never closed'."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw or None
    raw_str = str(raw).strip()
    if not raw_str or raw_str.lower() in ("unknown", "n/a", "na", "not specified"):
        return None
    days = [d.strip().title() for d in re.split(r"[,/;]", raw_str) if d.strip()]
    return days or None


# --------------------------------------------------------------------------
# Google Places -> NormalizedPlace
# --------------------------------------------------------------------------

# Google's business_status values map directly; openNow is a live boolean.
def from_google_place(raw: dict) -> NormalizedPlace:
    """
    `raw` is a Place object from Places API (New) Text/Nearby Search,
    already restricted to the field mask we requested (see
    retrieval/google_places.py). Only fields present in our field mask are
    read here — anything else is legitimately absent, not "missing data".
    """
    location = raw.get("location") or {}
    opening = raw.get("currentOpeningHours") or {}
    price_level_map = {
        "PRICE_LEVEL_FREE": 0,
        "PRICE_LEVEL_INEXPENSIVE": 1,
        "PRICE_LEVEL_MODERATE": 2,
        "PRICE_LEVEL_EXPENSIVE": 3,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    }
    price_level = price_level_map.get(raw.get("priceLevel")) if raw.get("priceLevel") else None

    display_name = (raw.get("displayName") or {}).get("text") or raw.get("name") or "Unknown place"

    return NormalizedPlace(
        id=f"google:{raw.get('id')}",
        source="google",
        source_id=raw.get("id", ""),
        name=display_name,
        address=raw.get("formattedAddress"),
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        categories=raw.get("types") or ([raw["primaryType"]] if raw.get("primaryType") else []),
        tags=[],
        description=(raw.get("editorialSummary") or {}).get("text"),
        # Google does not give us a per-person cost figure — priceLevel is a
        # 0-4 relative indicator, not an INR amount. We surface it via
        # price_level and leave Cost unknown rather than inventing a number.
        cost=Cost(basis="unknown", is_free=(price_level == 0) if price_level is not None else None),
        capacity=Capacity(),  # Google has no capacity concept for a place
        duration_minutes=None,  # Google does not provide a "visit duration" field
        rating=raw.get("rating"),
        review_count=raw.get("userRatingCount"),
        price_level=price_level,
        opening_hours=OpeningHours(
            weekly_closure=None,  # Google gives live open/closed, not a fixed weekly-closure list
            open_now=opening.get("openNow"),
            hours_text="; ".join(opening.get("weekdayDescriptions", [])) or None,
        ),
        business_status=raw.get("businessStatus"),
        local=LocalKnowledge(),  # Google engine never has local-knowledge-layer data
        last_verified=None,  # Google data is live-fetched, not "verified as of" a date
        source_url=raw.get("googleMapsUri"),
    )


# --------------------------------------------------------------------------
# Mumbai dataset -> NormalizedPlace
# --------------------------------------------------------------------------

def from_mumbai_record(raw: dict) -> NormalizedPlace:
    """
    `raw` is one row from data/mumbai_places.json (or the MySQL
    `mumbai_places` table via retrieval/mysql_retriever.py — same shape).
    Every field is read with .get() and passed through the parse_* helpers
    so an absent column becomes an explicit None, never a fabricated value.
    """
    categories = raw.get("category") or []
    if isinstance(categories, str):
        categories = [categories]
    sub = raw.get("sub_category")
    if sub:
        categories = categories + ([sub] if isinstance(sub, str) else list(sub))

    tags = raw.get("experience_tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    traveller_tags = raw.get("traveller_tags") or []
    if isinstance(traveller_tags, str):
        traveller_tags = [t.strip() for t in traveller_tags.split(",") if t.strip()]

    amount = raw.get("est_cost_per_person_inr")
    cost_basis_text = raw.get("cost_basis") if isinstance(raw.get("cost_basis"), str) else None
    if amount is not None:
        # A numeric estimate exists — trust it, and keep any free-text note alongside.
        cost = parse_cost(amount=amount, cost_basis="per_person")
        if cost_basis_text:
            cost = cost.model_copy(update={"note": cost_basis_text})
    elif cost_basis_text:
        # No numeric estimate — parse the qualifier (e.g. "Free entry; spend varies")
        # instead of guessing a number.
        cost = parse_cost(free_text=cost_basis_text)
    else:
        cost = Cost()

    return NormalizedPlace(
        id=f"mumbai:{raw.get('provider_id')}",
        source="mumbai",
        source_id=str(raw.get("provider_id", "")),
        name=raw.get("provider_name", "Unknown provider"),
        address=raw.get("address"),
        latitude=raw.get("latitude"),
        longitude=raw.get("longitude"),
        categories=categories,
        tags=list(tags) + list(traveller_tags),
        description=raw.get("description"),
        cost=cost,
        capacity=parse_capacity(raw.get("people_per_experience")),
        duration_minutes=raw.get("recommended_duration"),
        rating=raw.get("rating"),
        review_count=raw.get("review_count"),
        price_level=None,
        opening_hours=OpeningHours(
            weekly_closure=parse_weekly_closure(raw.get("weekly_closure")),
            open_now=None,  # Mumbai dataset is static/curated, not live — never guess "open now"
            hours_text=raw.get("opening_hours"),
        ),
        business_status=None,
        local=LocalKnowledge(
            local_score=raw.get("local_score"),
            local_confidence=raw.get("local_confidence"),
            local_recommendation_count=raw.get("local_recommendation_count"),
            local_tips=raw.get("local_tips") or [],
        ),
        last_verified=raw.get("last_verified"),
        source_url=raw.get("source_url"),
    )


def build_embedding_text(place: NormalizedPlace) -> str:
    """
    The text representation actually embedded (Section 13). Concatenates
    name, category, tags, description and location context — NOT just the
    name — so "Colaba Causeway Market" and a query about "local shopping
    and handicrafts around Colaba" land close together in embedding space
    even though the words barely overlap.
    """
    parts = [place.name]
    if place.categories:
        parts.append("Category: " + ", ".join(place.categories) + ".")
    if place.tags:
        parts.append("Tags: " + ", ".join(place.tags) + ".")
    if place.description:
        parts.append(place.description)
    if place.local.local_tips:
        parts.append("Local tips: " + " ".join(place.local.local_tips))
    if place.address:
        parts.append("Location: " + place.address + ".")
    return " ".join(p.strip() for p in parts if p and p.strip())
