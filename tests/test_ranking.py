from datetime import datetime, timezone

from models.schemas import Budget, BudgetBasis, Capacity, Cost, Intent, NormalizedPlace, OpeningHours
from ranking.scoring import check_hard_constraints


def _place(**overrides) -> NormalizedPlace:
    base = dict(
        id="mumbai:TEST-1", source="mumbai", source_id="TEST-1", name="Test Place",
        latitude=19.0, longitude=72.8,
    )
    base.update(overrides)
    return NormalizedPlace(**base)


def test_permanently_closed_is_hard_excluded():
    place = _place(business_status="CLOSED_PERMANENTLY")
    intent = Intent(raw_query="x")
    assert check_hard_constraints(intent, place) is not None


def test_closed_today_is_hard_excluded():
    place = _place(opening_hours=OpeningHours(weekly_closure=["Monday"]))
    intent = Intent(raw_query="x")
    monday = datetime(2026, 9, 21, tzinfo=timezone.utc)  # a known Monday
    assert check_hard_constraints(intent, place, now=monday) is not None


def test_open_on_a_different_day_passes():
    place = _place(opening_hours=OpeningHours(weekly_closure=["Monday"]))
    intent = Intent(raw_query="x")
    tuesday = datetime(2026, 9, 22, tzinfo=timezone.utc)
    assert check_hard_constraints(intent, place, now=tuesday) is None


def test_group_too_large_for_capacity_is_hard_excluded():
    place = _place(capacity=Capacity(min_people=1, max_people=4))
    intent = Intent(raw_query="x", group_size=6)
    assert check_hard_constraints(intent, place) is not None


def test_group_within_capacity_passes():
    place = _place(capacity=Capacity(min_people=1, max_people=4))
    intent = Intent(raw_query="x", group_size=3)
    assert check_hard_constraints(intent, place) is None


def test_unknown_capacity_never_excludes():
    place = _place(capacity=Capacity())  # unknown
    intent = Intent(raw_query="x", group_size=20)
    assert check_hard_constraints(intent, place) is None
