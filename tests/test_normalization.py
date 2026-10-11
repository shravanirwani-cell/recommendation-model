from normalization.normalized_place import parse_capacity, parse_cost, parse_weekly_closure


def test_capacity_range_parses_as_min_max_not_average():
    cap = parse_capacity("1-4")
    assert cap.min_people == 1
    assert cap.max_people == 4  # NOT interpreted as "typically 1"


def test_capacity_none_is_unknown():
    cap = parse_capacity(None)
    assert cap.min_people is None
    assert cap.max_people is None


def test_free_entry_spend_varies_is_not_zero_cost():
    cost = parse_cost(free_text="Free entry; spend varies")
    # Per Section 18: this must NOT collapse to "the whole experience is ₹0".
    assert cost.is_free is None  # ambiguous/mixed, not True
    assert cost.min_inr is None
    assert cost.max_inr is None


def test_unambiguous_free_entry():
    cost = parse_cost(free_text="Free entry to view; no ticket required")
    assert cost.is_free is True


def test_numeric_cost_estimate():
    cost = parse_cost(amount=500, cost_basis="per_person")
    assert cost.min_inr == 500
    assert cost.is_free is False


def test_weekly_closure_none_means_unknown_not_never_closed():
    assert parse_weekly_closure(None) is None
    assert parse_weekly_closure("") is None
    assert parse_weekly_closure("Not specified") is None


def test_weekly_closure_parses_days():
    assert parse_weekly_closure("Monday") == ["Monday"]
    assert parse_weekly_closure("Monday, Tuesday") == ["Monday", "Tuesday"]
