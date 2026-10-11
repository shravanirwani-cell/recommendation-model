from intent.gemini_intent import extract_intent_rule_based


def test_extracts_mumbai_demo_query():
    q = ("We are 3 people with ₹2000 each. We have around 2 hours and want "
         "local shopping and handicrafts around Colaba.")
    intent = extract_intent_rule_based(q)

    assert intent.city == "Mumbai"
    assert intent.location_text == "Colaba"
    assert set(intent.categories) == {"shopping", "handicrafts"}
    assert intent.budget is not None
    assert intent.budget.amount == 2000
    assert intent.budget.basis.value == "per_person"
    assert intent.group_size == 3
    assert intent.duration_minutes == 120
    assert "local" in intent.preferences


def test_extracts_google_demo_query():
    q = "Find good cafes near Bangalore airport for 2 people."
    intent = extract_intent_rule_based(q)

    assert intent.city is None  # not a known Mumbai area -> Google engine
    assert intent.location_text == "Bangalore airport"
    assert intent.categories == ["cafe"]
    assert intent.group_size == 2
    assert intent.budget is None


def test_missing_fields_stay_none_not_guessed():
    intent = extract_intent_rule_based("Show me something fun to do.")
    assert intent.budget is None
    assert intent.group_size is None
    assert intent.duration_minutes is None
    assert intent.location_text is None


def test_total_vs_per_person_budget_basis():
    intent = extract_intent_rule_based("We have a total budget of ₹5000 for the day.")
    assert intent.budget.amount == 5000
    assert intent.budget.basis.value == "total"
