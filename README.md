# TravelSATHI Recommendation Service

Standalone retrieval + ranking microservice for TravelSATHI. Owned entirely
by the ML/Recommendation Systems role — no React Native, no changes to the
main backend. See `DESIGN.md` for the full architecture writeup, risks, and
the rationale behind every design decision; this file is the practical
"how to run it and call it" reference.

## What this is

Two recommendation engines behind one API:

- **Mumbai engine** — curated local dataset (`data/mumbai_places.json`),
  richer fields, no external API dependency.
- **Google engine** — general-purpose, any city, backed by Google Places
  API (New) Text Search.

Both funnel through the *same* retrieval→constraints→semantic→ranking→top-K
pipeline and return the *same* response shape (see "Response shape" below).
Full rationale in `DESIGN.md`.

## Quickstart

```bash
cd recommendation_service
python -m venv .venv && source .venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt
cp .env.example .env        # optional — every value has a safe default
uvicorn app:app --reload --port 8000
```

That's it — **no API keys required to run it.** Without `GEMINI_API_KEY`,
intent extraction uses a deterministic rule-based extractor. Without
`GOOGLE_PLACES_API_KEY`, the Google engine returns clearly-labeled `[MOCK]`
candidates so the full pipeline is still demoable. Add either key later (in
`.env`) and nothing else changes — the service automatically switches to the
real thing.

Run the tests (no network, no keys needed):
```bash
pytest
```

## Endpoints (Section 21)

| Endpoint | Accepts | Returns |
|---|---|---|
| `POST /recommend` | `RecommendRequest` (`mode="auto"` picks Mumbai vs Google automatically from the query) | `RecommendResponse` |
| `POST /recommend/mumbai` | same request shape, engine forced to Mumbai | `RecommendResponse` |
| `POST /recommend/google` | same request shape, engine forced to Google | `RecommendResponse` |
| `POST /feedback` | `FeedbackEvent` | `{"status": "recorded"}` |
| `POST /embed/place` | `{"text": "..."}` (debug/ops only) | raw embedding vector |
| `GET /health` | — | `{"status": "ok", "embedding_backend": "st"|"hashing"}` |

## Request shape

```json
{
  "query": "We are 3 people with ₹2000 each. We have around 2 hours and want local shopping and handicrafts around Colaba.",
  "mode": "auto",
  "user_id": "user_123",
  "latitude": null,
  "longitude": null,
  "top_k": 2
}
```

`mode` is one of `"auto"` (default), `"mumbai"`, `"google"`. `latitude`/
`longitude`, if the app has GPS, override whatever intent extraction guessed
from the text.

## Response shape (Section 4/20 — real output, `top_k=2`)

```json
{
  "query_id": "Q-35b48e580501",
  "mode": "mumbai",
  "intent": {
    "location_text": "Colaba",
    "city": "Mumbai",
    "categories": ["shopping", "handicrafts"],
    "budget": {"amount": 2000.0, "currency": "INR", "basis": "per_person", "is_max": true},
    "group_size": 3,
    "duration_minutes": 120,
    "preferences": ["local"],
    "extraction_method": "rule_based",
    "confidence": 0.55
  },
  "recommendations": [
    {
      "place_id": "mumbai:TS-MUM-001",
      "source": "mumbai",
      "name": "Colaba Causeway Market",
      "address": "Shahid Bhagat Singh Road, Colaba, Mumbai",
      "latitude": 18.907, "longitude": 72.8115,
      "category": ["shopping", "markets", "street market"],
      "distance_km": 0.34,
      "estimated_cost_per_person": null,
      "cost_note": "Free entry; spend varies",
      "rating": null,
      "opening_status": "unknown",
      "semantic_score": 0.6652,
      "constraint_score": 0.6416,
      "final_score": 0.6511,
      "match_reasons": [
        "Matches shopping preference",
        "Located in/near Colaba",
        "Close to your specified location"
      ],
      "score_breakdown": { "...": "full per-feature breakdown, see below" }
    }
  ],
  "engine_meta": {
    "candidates_considered": 4,
    "candidates_after_constraints": 4,
    "engine": "mumbai",
    "data_source": "json/mysql",
    "warnings": []
  },
  "generated_at": 1790417217.08
}
```

The full response (including each recommendation's complete `details` block
— the entire `NormalizedPlace`, useful for a detail screen) is in
`sample_requests/mumbai_demo_response.json`, generated from an actual run of
this exact code, not hand-written. A second full example for the Google
engine's demo query is in `sample_requests/google_demo_response.json`
(mock-mode candidates, since no `GOOGLE_PLACES_API_KEY` is configured in
this repo).

**Every field your MySQL backend needs to persist/pass through to React
Native is a flat top-level field** (`place_id`, `name`, `latitude`,
`longitude`, `final_score`, etc.) — `details` is optional extra depth, not
required for the basic list view.

## Feedback events (Section 22)

```json
POST /feedback
{
  "user_id": "user_123",
  "query_id": "Q-35b48e580501",
  "place_id": "mumbai:TS-MUM-001",
  "event": "click"
}
```

`event` ∈ `impression | click | save | share | booking | visit | rating |
reject`. For `event="rating"`, also send `"value": 4.5`. Stored to
`data/feedback_log.jsonl` by default, or into MySQL's
`recommendation_feedback` table if `FEEDBACK_STORE=mysql` (see
`sql/schema.sql`). This is a fire-and-forget call — it never blocks or fails
a recommendation request.

---

## HOW MY TEAMMATES INTEGRATE THIS

You do not need to understand any ML internals. Your interaction is exactly:

```
React Native → Your existing backend → POST /recommend → this service → JSON → your backend → React Native
```

### 1. What you run

Nothing, if you don't want to — the ML engineer runs this service as its own
process (locally, or containerized later). Your backend just needs its base
URL (e.g. `http://localhost:8000` in dev, or wherever it's deployed).

If you ever need to run it yourself:
```bash
cd recommendation_service
pip install -r requirements.txt
uvicorn app:app --port 8000
```

### 2. Environment variables you provide (optional — all have safe defaults)

None are required from you. If/when you want the Mumbai dataset to live in
*your* MySQL database instead of the bundled JSON file, set:
```
MUMBAI_DATA_SOURCE=mysql
MYSQL_HOST=... MYSQL_PORT=... MYSQL_USER=... MYSQL_PASSWORD=... MYSQL_DATABASE=...
```
and run `sql/schema.sql` against your database once. Same for
`FEEDBACK_STORE=mysql` if you want feedback events in MySQL instead of the
local JSONL file.

### 3. Which endpoint you call

`POST {SERVICE_URL}/recommend` for the normal case (it auto-picks Mumbai vs
Google). Use `/recommend/mumbai` or `/recommend/google` only if you need to
force one engine regardless of what the user typed.

### 4. What JSON you send

```json
{
  "query": "<the user's free-text request, verbatim>",
  "mode": "auto",
  "user_id": "<your user id, or omit>",
  "latitude": <optional, if you have GPS>,
  "longitude": <optional>,
  "top_k": 10
}
```

### 5. What JSON you receive

See "Response shape" above. The one field you MUST store/pass along
unchanged is `query_id` — you'll need it if you ever send feedback events
back (step 6), and it's how a future analytics/ranking-improvement pass ties
a click back to the query that produced it.

### 6. How you store feedback

Whenever your app already knows a user did something with a recommended
place (tapped it, saved it, booked it, rated it), send it here — you don't
need to build your own separate feedback pipeline:
```
POST {SERVICE_URL}/feedback
{ "user_id": "...", "query_id": "...", "place_id": "...", "event": "click" }
```
Fire-and-forget from your backend; don't block the user-facing response on
it, and don't retry aggressively if it fails — it's non-critical telemetry.

### 7. How React Native receives it

Unchanged from your existing pattern — your backend receives the JSON from
step 5, stores/logs whatever it needs (e.g. `query_id`, place IDs shown, for
step 6), and forwards the `recommendations` array (or a trimmed version of
it) to the app exactly like any other API response you already return.

**That's the entire integration.** Everything else in this repository
(embeddings, ranking weights, Google Places quirks, the Mumbai dataset
schema) is this module's internal concern and can change without affecting
your side of the contract, as long as the request/response JSON shapes above
stay the same.

---

## Repository layout

```
recommendation_service/
├── app.py                    FastAPI HTTP layer (thin — see service.py for the real logic)
├── service.py                Framework-agnostic pipeline entrypoint (run_recommendation)
├── config.py                 All environment-variable configuration, one place
├── requirements.txt
├── .env.example
├── DESIGN.md                 Full architecture writeup (read this for "why")
├── README.md                 This file
├── intent/
│   └── gemini_intent.py      Gemini structured extraction + rule-based fallback
├── retrieval/
│   ├── google_places.py      Engine A: Places API (New) Text Search
│   ├── mumbai_retriever.py   Engine B: JSON or MySQL-backed
│   ├── mysql_retriever.py    Optional MySQL reads for the Mumbai dataset
│   └── candidate_retriever.py  Engine routing (auto/mumbai/google)
├── normalization/
│   └── normalized_place.py   Raw source record -> NormalizedPlace (+ Section 18 parsing)
├── embeddings/
│   ├── embedder.py            sentence-transformers, auto-falls back to TF-IDF/SVD
│   └── vector_store.py        In-memory cosine similarity search
├── ranking/
│   ├── feature_engineering.py  Per-feature scores (distance/budget/time/group/quality/local)
│   ├── scoring.py              Hard constraints + weighted final_score + match_reasons
│   └── ranker.py                Orchestrates Stages 2-5
├── models/
│   └── schemas.py              Every Pydantic contract (Intent, NormalizedPlace, Recommendation, ...)
├── feedback/
│   └── feedback_store.py       JSON or MySQL feedback event storage
├── data/
│   ├── mumbai_places.json      Seed dataset (10 real, verified-public-landmark records)
│   └── feedback_log.jsonl      Created at runtime (json feedback store)
├── sql/
│   └── schema.sql               Optional MySQL DDL (mumbai_places, recommendation_feedback, google_place_id_cache)
├── scripts/
│   └── build_embeddings.py      Offline batch embedding generation for the Mumbai dataset
├── sample_requests/              Real, generated request/response examples
└── tests/                        pytest suite — no network, no API keys needed
```
