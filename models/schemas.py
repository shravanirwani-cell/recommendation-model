"""
Shared Pydantic schemas for the TravelSATHI recommendation service.

These are the contracts everything else in the service (and your
teammates' backend) is built against:

    RecommendRequest   -> what the backend sends to POST /recommend
    Intent             -> structured interpretation of the user's query
    NormalizedPlace    -> the single internal shape BOTH engines produce
    Recommendation     -> one ranked result (NormalizedPlace + scores)
    RecommendResponse  -> what POST /recommend returns
    FeedbackEvent      -> what the backend sends to POST /feedback

Design notes (see DESIGN.md Step 2 for the full rationale):
  * Every numeric field that could be "unknown" is Optional and defaults to
    None — never 0 or "" — because 0/"" are valid real values (e.g. a free
    experience really can have cost=0) and we must never conflate "unknown"
    with "zero" (Section 18 of the spec).
  * cost/capacity are small structured objects, not bare numbers, because a
    single number cannot represent "1-4 people" or "free entry, spend
    varies" without losing information or fabricating precision.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Intent (Section 12)
# --------------------------------------------------------------------------

class BudgetBasis(str, Enum):
    PER_PERSON = "per_person"
    TOTAL = "total"
    UNKNOWN = "unknown"


class Budget(BaseModel):
    amount: Optional[float] = Field(None, description="Numeric budget figure as stated by the user.")
    currency: str = "INR"
    basis: BudgetBasis = BudgetBasis.UNKNOWN
    is_max: bool = Field(True, description="True if this is an upper bound (almost always is for travellers).")


class Intent(BaseModel):
    """Structured interpretation of a free-text travel request."""

    raw_query: str
    location_text: Optional[str] = Field(None, description="Location phrase as stated, e.g. 'Colaba, Mumbai'.")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = Field(None, description="Search radius; filled with a default if unspecified.")
    city: Optional[str] = Field(None, description="Resolved city, used to pick Mumbai engine vs Google engine.")

    categories: list[str] = Field(default_factory=list)
    budget: Optional[Budget] = None
    group_size: Optional[int] = None
    duration_minutes: Optional[int] = Field(None, description="User's AVAILABLE time — not a place's visit duration.")
    traveller_type: Optional[str] = Field(None, description="e.g. 'family', 'solo', 'couple', 'friends'.")
    preferences: list[str] = Field(default_factory=list, description="Soft qualifiers, e.g. 'local', 'authentic'.")
    time_constraint: Optional[str] = Field(None, description="e.g. 'open now', 'evening', 'weekend'.")

    extraction_method: Literal["gemini", "rule_based"] = "rule_based"
    confidence: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("categories", "preferences", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return v or []


# --------------------------------------------------------------------------
# Normalized place (Section 2 / 17) — the single shape both engines emit
# --------------------------------------------------------------------------

class Cost(BaseModel):
    """Represents cost without pretending to know more than the source does."""

    min_inr: Optional[float] = None
    max_inr: Optional[float] = None
    basis: BudgetBasis = BudgetBasis.UNKNOWN
    is_free: Optional[bool] = Field(None, description="None = unknown, NOT False.")
    note: Optional[str] = Field(None, description="Raw qualifier, e.g. 'Free entry; spend varies'.")

    @property
    def is_known(self) -> bool:
        return self.min_inr is not None or self.max_inr is not None or self.is_free is not None


class Capacity(BaseModel):
    """Represents group-size capacity, e.g. '1-4' -> min=1, max=4 (max, not typical)."""

    min_people: Optional[int] = None
    max_people: Optional[int] = None
    raw: Optional[str] = None


class OpeningHours(BaseModel):
    """weekly_closure=None means UNKNOWN, never 'open every day'."""

    weekly_closure: Optional[list[str]] = Field(None, description="Days closed, e.g. ['Monday']. None = unknown.")
    open_now: Optional[bool] = None
    hours_text: Optional[str] = Field(None, description="Human-readable hours if we have them.")


class LocalKnowledge(BaseModel):
    """Section 11 — future crowdsourced signal. Always optional, never load-bearing."""

    local_score: Optional[float] = None
    local_confidence: Optional[float] = Field(None, description="How many contributors back local_score.")
    local_recommendation_count: Optional[int] = None
    local_tips: list[str] = Field(default_factory=list)


class NormalizedPlace(BaseModel):
    """
    The one internal representation both the Mumbai engine and the Google
    Places engine convert their raw records into, BEFORE ranking. The
    ranking pipeline (ranking/) only ever looks at this shape — it never
    knows whether a place came from Google or from the Mumbai dataset.
    """

    id: str
    source: Literal["google", "mumbai"]
    source_id: str = Field(..., description="place_id (Google) or provider_id (Mumbai) — for re-fetching / audit.")

    name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    cost: Cost = Field(default_factory=Cost)
    capacity: Capacity = Field(default_factory=Capacity)
    duration_minutes: Optional[int] = Field(None, description="PROVIDER's estimated visit duration, if known.")

    rating: Optional[float] = None
    review_count: Optional[int] = None
    price_level: Optional[int] = Field(None, description="0-4, Google-style scale, when available.")

    opening_hours: OpeningHours = Field(default_factory=OpeningHours)
    business_status: Optional[str] = None

    local: LocalKnowledge = Field(default_factory=LocalKnowledge)

    last_verified: Optional[str] = Field(None, description="ISO date the source data was last checked/refreshed.")
    source_url: Optional[str] = None

    embedding_text: Optional[str] = Field(None, description="The text actually embedded — kept for debuggability.")

    class Config:
        extra = "allow"  # future fields (Section 10) don't break the schema


# --------------------------------------------------------------------------
# Ranking / scoring
# --------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    semantic_score: Optional[float] = None
    distance_score: Optional[float] = None
    budget_score: Optional[float] = None
    time_score: Optional[float] = None
    group_score: Optional[float] = None
    quality_score: Optional[float] = None
    local_score: Optional[float] = None
    final_score: float = 0.0


class Recommendation(BaseModel):
    """One ranked result — flattened for easy consumption by React Native / MySQL."""

    place_id: str
    source: Literal["google", "mumbai"]
    name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    category: list[str] = Field(default_factory=list)

    distance_km: Optional[float] = None
    estimated_cost_per_person: Optional[float] = None
    cost_note: Optional[str] = None
    rating: Optional[float] = None
    opening_status: Optional[str] = Field(None, description="'open' | 'closed' | 'unknown'.")

    semantic_score: Optional[float] = None
    constraint_score: Optional[float] = None
    final_score: float = 0.0

    match_reasons: list[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    details: dict[str, Any] = Field(default_factory=dict, description="Full NormalizedPlace, for detail views.")


class EngineMeta(BaseModel):
    candidates_considered: int = 0
    candidates_after_constraints: int = 0
    engine: Literal["google", "mumbai"]
    data_source: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text natural language request.")
    mode: Literal["auto", "mumbai", "google"] = "auto"
    user_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)


class RecommendResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: new_id("Q"))
    mode: Literal["mumbai", "google"]
    intent: Intent
    recommendations: list[Recommendation] = Field(default_factory=list)
    engine_meta: EngineMeta
    generated_at: float = Field(default_factory=time.time)


# --------------------------------------------------------------------------
# Feedback (Section 22)
# --------------------------------------------------------------------------

class FeedbackEventType(str, Enum):
    IMPRESSION = "impression"
    CLICK = "click"
    SAVE = "save"
    SHARE = "share"
    BOOKING = "booking"
    VISIT = "visit"
    RATING = "rating"
    REJECT = "reject"


class FeedbackEvent(BaseModel):
    user_id: Optional[str] = None
    query_id: str
    place_id: str
    event: FeedbackEventType
    value: Optional[float] = Field(None, description="Used for event='rating' (e.g. 1-5). Ignored otherwise.")
    dwell_time_seconds: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)
