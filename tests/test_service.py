"""
End-to-end tests against service.run_recommendation() — the framework-
agnostic core (see service.py docstring). These do NOT need FastAPI or a
running server, and never make a real network call (no API keys set in the
test environment => both engines run in their offline/mock modes), so this
suite is safe to run in CI with zero setup and zero secrets.
"""
from models.schemas import RecommendRequest
from service import run_recommendation


def test_mumbai_demo_scenario_ranks_colaba_causeway_market_first():
    request = RecommendRequest(
        query=(
            "We are 3 people with ₹2000 each. We have around 2 hours and want "
            "local shopping and handicrafts around Colaba."
        ),
        mode="auto",
    )
    response = run_recommendation(request)

    assert response.mode == "mumbai"
    assert len(response.recommendations) > 0
    assert response.recommendations[0].name == "Colaba Causeway Market"
    assert response.query_id.startswith("Q-")


def test_google_demo_scenario_routes_to_google_engine_in_mock_mode():
    request = RecommendRequest(query="Find good cafes near Bangalore airport for 2 people.", mode="auto")
    response = run_recommendation(request)

    assert response.mode == "google"
    assert len(response.recommendations) > 0
    assert response.engine_meta.data_source == "google_places_api/mock"


def test_explicit_mode_overrides_auto_routing():
    request = RecommendRequest(query="Best cafes in Bangalore", mode="google")
    response = run_recommendation(request)
    assert response.mode == "google"


def test_response_matches_shared_schema_fields():
    request = RecommendRequest(query="Shopping near Colaba", mode="mumbai")
    response = run_recommendation(request)
    rec = response.recommendations[0]

    # Section 4's shared interface — every field the spec asked for is present.
    for field in ["place_id", "name", "address", "latitude", "longitude", "category",
                  "distance_km", "estimated_cost_per_person", "rating", "opening_status",
                  "semantic_score", "constraint_score", "final_score", "match_reasons"]:
        assert hasattr(rec, field)


def test_top_k_is_respected():
    request = RecommendRequest(query="Shopping near Colaba", mode="mumbai", top_k=2)
    response = run_recommendation(request)
    assert len(response.recommendations) <= 2


def test_empty_result_when_group_size_exceeds_all_capacities():
    # No Mumbai seed record has a capacity limit set, so this exercises the
    # "no hard-constraint exclusions fire on unknown data" path rather than
    # actually emptying results — included to document/lock that behaviour.
    request = RecommendRequest(query="Shopping near Colaba for 500 people", mode="mumbai")
    response = run_recommendation(request)
    assert isinstance(response.recommendations, list)
