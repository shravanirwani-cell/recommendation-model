# TravelSATHI Recommendation Service — Design Document

Owner: ML / Recommendation Systems (this module). Scope: retrieval, ranking,
semantic search, ML, and the recommendation-service API contract only — no
React Native, no rewriting the main backend. This document follows the
STEP 1–13 order requested, plus the two closing sections (team integration,
demo script) as their own explicitly-labeled parts.

---

## STEP 1 — Architecture analysis & risks

**Chosen shape:** a standalone Python service (FastAPI) that the team's
existing MySQL/REST backend calls over HTTP, per the diagram in the original
brief. This is Option B (FastAPI microservice) — see the comparison below.

### A vs B vs C (Section 5)

| Option | What it means | Verdict for this hackathon |
|---|---|---|
| **A. Python module imported into backend** | Your teammates' backend directly `import`s your Python package. | Rejected: their backend's stack/deploy target is theirs to own; forcing a Python import into it (if it's Node/Java/etc, this is impossible anyway) creates tight coupling and makes your ML dependencies (numpy, sklearn, sentence-transformers) their problem too. |
| **B. FastAPI microservice** | Your service runs as its own process; they call `POST /recommend` over HTTP. | **Chosen.** Clean separation of concerns, matches "standalone module my teammates can call from their existing backend" exactly as asked, trivial to run locally (`uvicorn app:app`) or containerize later, and the only integration surface your teammates need is one HTTP call + one JSON contract (README.md). |
| **C. Standalone ML service (e.g. behind a queue, separately scaled)** | Overkill infra (message queues, separate scaling policies) for a hackathon. | Rejected for now — nothing here needs async job processing; a request/response HTTP call is enough. Revisit only post-hackathon if traffic demands it. |

### Key risks and how the design addresses them

1. **Google Places terms risk** — accidentally building a permanent local
   copy of "all of Google Places" (the spec explicitly warns against this).
   *Mitigation:* `retrieval/google_places.py` only ever caches results
   in-memory with a TTL (`GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS`, default 1
   hour) and never writes non-`place_id` Google content to disk/MySQL. See
   "Google Places data policy" below.
2. **No real Mumbai dataset yet** — `data/mumbai_places.json` is a **10-place
   seed dataset of real, well-known public landmarks only** (Colaba
   Causeway Market, Crawford Market, Gateway of India, etc.), with every
   uncertain numeric field (cost, rating, duration, capacity) left `null`
   rather than invented, per the project's no-fabrication rule. This is
   enough to demo both engines end-to-end, but restaurants/cafes/individual
   vendors/tour operators are deliberately **not** included — naming a
   specific business with a fabricated price or rating would violate that
   rule. Real curated research is a prerequisite before this dataset is
   demo-ready beyond the "market/heritage/landmark" categories.
3. **No API keys available in this dev environment** — `GEMINI_API_KEY`
   and `GOOGLE_PLACES_API_KEY` are both optional. Every external call has an
   automatic, clearly-logged fallback (rule-based intent extraction; mock
   Google Places candidates prefixed `[MOCK]`), so the full pipeline runs
   and is testable with zero keys, and upgrades to real data the moment keys
   are added — no code changes needed.
4. **No production sentence-transformers model available in this dev
   environment either** (no internet to download it here) — `embeddings/embedder.py`
   falls back to a TF-IDF+SVD embedder automatically. This is NOT a
   production-quality semantic model, but it exercises the exact same code
   path (`Embedder.encode()`), so ranking, tests, and the API contract are
   all verified against real logic, not stubs. On the team's dev machines
   (with normal internet access) it will automatically use real
   sentence-transformers embeddings instead — no config change needed.
5. **Provider-ID classification anti-pattern** — explicitly avoided. There is
   no `user -> place_id` classifier anywhere; every request re-runs
   retrieval + ranking over whatever candidates exist for that query
   (Section 7). This is a retrieval+ranking system, not a recommender that
   memorizes a fixed candidate list.
6. **Missing/ambiguous data silently becoming wrong data** — handled once,
   at the normalization boundary (`normalization/normalized_place.py`), not
   scattered through ranking logic. See Section 18 handling below.

---

## STEP 2 — Normalized place schema

Both engines emit `models.schemas.NormalizedPlace` — see that file for the
authoritative definition and inline rationale. Key improvements over the
brief's sketch:

- **`cost` and `capacity` are structured objects, not bare numbers** —
  `Cost{min_inr, max_inr, basis, is_free, note}` and
  `Capacity{min_people, max_people, raw}` — because a single float cannot
  represent "1–4 people" or "free entry, spend varies" without losing or
  fabricating information (Section 18's explicit examples).
- **Every uncertain field defaults to `None`, never `0`/`""`** — an unknown
  `weekly_closure` is `None`, not `[]` ("never closed"); an unknown cost is
  `Cost()` with `is_free=None`, not `is_free=False`.
- **`local: LocalKnowledge`** is its own nested, fully-optional block
  (Section 11) so the Local Knowledge Layer can be added later without a
  schema migration, and so ranking code can check `if place.local.local_score
  is not None` in one place.
- **`Config.extra = "allow"`** so future Mumbai-dataset columns (Section 10's
  "richer future fields") don't break deserialization before the schema is
  formally extended.
- **`embedding_text`** is stored on the object (not thrown away after
  embedding) specifically so `/embed/place` and logs can show *exactly* what
  was embedded — a debuggability requirement (Section 29 #23: "include
  logging so we can debug why a place received a particular score").

`Intent`, `Recommendation`, `RecommendResponse`, `FeedbackEvent` are the
other four schemas — see `models/schemas.py` docstrings for each.

---

## STEP 3 — Recommendation pipeline

Implemented exactly as the five stages in the brief, in `ranking/ranker.py`
(`rank_candidates()`), fed by `retrieval/candidate_retriever.py`:

```
Intent extraction  (intent/gemini_intent.py)
        |
Engine routing      (retrieval/candidate_retriever.resolve_engine)
        |
Stage 1  Candidate retrieval   (retrieval/google_places.py OR mumbai_retriever.py)
        |
Stage 2  Hard constraint filtering   (ranking/scoring.check_hard_constraints)
        |
Stage 3  Semantic relevance          (embeddings/*, cosine similarity)
        |
Stage 4  Hybrid ranking               (ranking/scoring.score_candidate)
        |
Stage 5  Top-K                        (ranking/ranker.rank_candidates, sort+slice)
```

Both engines converge on the exact same Stage 2–5 code
(`ranking/ranker.py` has no `if source == "google"` branches) — this is the
"one recommendation algorithm, not two unrelated systems" requirement
(Section 17).

---

## STEP 4 — Google Places retrieval design

### How it's queried (Section 8/9)

- **Endpoint:** `POST https://places.googleapis.com/v1/places:searchText`
  (Places API (New) Text Search) — confirmed against current official docs.
- **Field mask (mandatory header, `X-Goog-FieldMask`):** deliberately
  restricted to **Essentials + Pro + Enterprise** tiers — `id, displayName,
  formattedAddress, location, types, primaryType, businessStatus, rating,
  userRatingCount, priceLevel, currentOpeningHours, googleMapsUri`. This
  explicitly **excludes** `reviews`, `photos`, `editorialSummary` and other
  **Enterprise + Atmosphere** fields, which are billed at the most expensive
  tier and aren't needed for ranking.
- **Location bias:** a `circle` (`center` = intent lat/lon, `radius` = intent
  radius in meters) — biases, doesn't hard-restrict, results toward the
  area, matching "the system should work for arbitrary supported geographic
  locations."
- **`includedType`:** set only when the intent resolves to exactly one
  strong category with a clean Places-type mapping (e.g. `cafe` → `cafe`,
  `shopping` → `shopping_mall`) — see `CATEGORY_TO_TYPE` in
  `retrieval/google_places.py`. Left unset otherwise so the natural-language
  `textQuery` does the heavy lifting (Google's own ranking already combines
  text relevance with type well).
- **Candidate volume:** paginates (`pageSize=20`, following `nextPageToken`)
  up to `GOOGLE_PLACES_CANDIDATE_LIMIT` (default 60) — inside the spec's
  "20–100 candidates before expensive ranking" target, and short of Text
  Search's documented 60-results-across-all-pages ceiling.

### Google Places data policy (Section 9) — verified 2026-09-26

Sources: [Places API caching/storage policy](https://developers.google.com/maps/documentation/places/web-service/policies),
[Places API usage & billing](https://developers.google.com/maps/documentation/places/web-service/usage-and-billing),
[Text Search (New)](https://developers.google.com/maps/documentation/places/web-service/text-search),
[Place Data Fields (New)](https://developers.google.com/maps/documentation/places/web-service/data-fields).

| Data | May we store it permanently? |
|---|---|
| **`place_id` (the `id` field)** | **Yes — explicitly exempt.** Google's policy states place IDs may be stored indefinitely. This is why `sql/schema.sql` includes a place-ID-only `google_place_id_cache` table as the *only* thing safe to persist from Google. |
| Everything else (name, address, rating, hours, price level, etc.) | **No — not by default.** The policy prohibits pre-fetching/caching/storing Places content beyond documented exceptions. `retrieval/google_places.py` therefore only holds this in a **process-local, in-memory, TTL-bound** cache (`GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS`, default 1 hour) — it is never written to MySQL or disk. |
| **Pricing** | Places API (New) bills per SKU tier — **Essentials → Pro → Enterprise → Enterprise+Atmosphere** — and you're billed at the *highest* tier any field in your mask touches. Our field mask deliberately stays out of the most expensive (Enterprise+Atmosphere) tier. Exact current per-1,000-call rates are on Google's own pricing page/calculator (they change independently of this document) — check `usage-and-billing` before the hackathon if cost matters for your GCP budget. |
| **Quotas** | Configured per-project in Google Cloud Console; not a fixed public number this doc should assert. Check your GCP project's quota page directly. |

**Action item for whoever holds the Google Cloud project:** re-check the
`policies` page before demo day in case terms changed, and do not widen the
field mask or add a persistent (non-`place_id`) Google Places table without
re-reading it.

---

## STEP 5 — Mumbai database retrieval design

- **Source of truth today:** `data/mumbai_places.json` (10 seed records —
  see STEP 1 risk #2). Swap to MySQL later by setting
  `MUMBAI_DATA_SOURCE=mysql` — `retrieval/mysql_retriever.py` reads the same
  shape from a `mumbai_places` table (DDL in `sql/schema.sql`); no other code
  changes.
- **Candidate generation (Stage 1) is cheap and coarse on purpose:**
  `retrieval/mumbai_retriever.py` does a generous haversine radius filter
  (2.5× the requested radius — wide enough not to accidentally exclude a
  real match) and an optional category/tag keyword filter that **backs off
  to the wider set** if it would otherwise return zero candidates. Real
  precision comes from Stage 3 (semantic) and Stage 4 (ranking), not from
  aggressive pre-filtering — important while the dataset is small and every
  category isn't perfectly tagged yet.
- **Schema fields not present are simply absent, never guessed** — see
  `from_mumbai_record()` and the parsing helpers (`parse_capacity`,
  `parse_cost`, `parse_weekly_closure`) in `normalization/normalized_place.py`,
  which directly implement Section 18's three worked examples
  (`weekly_closure=null`, `"Free entry; spend varies"`, `"1-4"`) — see
  `tests/test_normalization.py` for tests asserting exactly this behaviour.

---

## STEP 6 — Embedding generation & vector search

### Text representation (Section 13)

`normalization.normalized_place.build_embedding_text()` concatenates name +
categories + tags + description + local tips + address — never just the
name — so `"Colaba Causeway Market"` and a query about `"local shopping and
handicrafts around Colaba"` land close in embedding space despite minimal
literal word overlap. Verified empirically: it ranks first for the spec's
own Demo 2 query (see `tests/test_service.py`).

### Model / storage comparison

| Option | Verdict |
|---|---|
| **Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim)** | **Chosen (primary).** Free, runs on CPU fast enough for a hackathon demo, no per-call cost, small download (~80MB), strong general semantic quality. |
| Gemini / Google embedding models | Viable alternative, adds an API-key dependency and per-call cost/latency for something that doesn't need an LLM. Kept as a documented option, not the default — sentence-transformers is simpler for a self-contained module your teammates just run. |
| **TF-IDF + TruncatedSVD (scikit-learn)** | **Chosen (automatic fallback)** when sentence-transformers isn't installed or can't download its model (e.g. offline dev/CI). Zero network dependency, deterministic, but weaker semantically (won't bridge unseen synonyms). This is what actually ran during this project's own development/testing sandbox — see `embeddings/embedder.py`. |
| **FAISS / Qdrant / Pinecone** (vector databases) | **Not used.** Candidate sets are 20–100 places per request by design (Section 8) — brute-force numpy cosine similarity over that many 384-dim vectors is sub-millisecond. A vector DB adds infrastructure with no latency benefit at this scale. See `embeddings/vector_store.py` docstring for the explicit "graduate to FAISS when the Mumbai dataset passes ~5–10k rows" trigger. |
| **pgvector** | Would make sense if the team's MySQL were Postgres; it's MySQL, so this was never in contention. Noted for completeness since the brief asked about it. |

### Practicals

- **Dimensions:** 384 (sentence-transformers) or 256 (SVD fallback, tunable).
- **Batch generation / updating:** `scripts/build_embeddings.py` — run
  manually after editing the Mumbai dataset; writes
  `data/embedding_cache.json`. Not yet wired into the live request path
  (today's dataset is cheap enough to re-embed per request); see the
  script's docstring for exactly what to change when the dataset grows.
- **Metadata filtering:** done *before* embedding (geo + category filter in
  `mumbai_retriever.py` / Google's own `includedType` + location bias), not
  via a vector DB's metadata filter — there's no vector DB in this design.
- **Storage:** none required today (query-time, in-memory only). The cache
  file above is the seed for future storage once retrieval graduates to
  reading pre-computed vectors instead of encoding on every request.

---

## STEP 7 — Hybrid ranking design

### Hard constraints vs soft/ranking features (Section 15)

| Category | Fields | Why |
|---|---|---|
| **Hard constraints** (candidate dropped entirely — `ranking/scoring.check_hard_constraints`) | `business_status` permanently/temporarily closed; `weekly_closure` includes today (only when *known*); `group_size > capacity.max_people` (only when *known*); `time_constraint="open now"` and `open_now is False` (only when *known*) | Each is a deterministic, unambiguous disqualifier *when the data is present*. Critically, **unknown never excludes** — `tests/test_ranking.py::test_unknown_capacity_never_excludes` locks this in. |
| **Soft/ranking features** (penalize via score, `ranking/feature_engineering.py`) | `distance_score`, `budget_score`, `time_score`, `group_score` (the *degree* of fit, once past the hard check), `quality_score`, `local_score` | Real-world data here is frequently partial or approximate (an estimated cost, a fuzzy duration) — a hard cutoff on approximate data would wrongly discard good matches. |
| **Semantic feature** | `semantic_score` | Cosine similarity between query and place embedding text — the one feature that isn't a structured attribute. |
| **Deliberately NOT a separate weight** | `category_score` | Category/tag terms are already part of `embedding_text`, so a separate category weight would double-count the same signal `semantic_score` already captures. |
| **Unknown feature handling** | any soft feature `None` | Substituted with a neutral `0.5` in the weighted sum (`ranking/scoring.score_candidate`) — pulls a candidate to the middle of the pack instead of being punished (0) or rewarded (1) for something genuinely unknown. |

### Starting weights (Section 15, configurable via env vars — see `.env.example`)

```
final_score =
    0.40 * semantic_score
  + 0.20 * distance_score
  + 0.15 * budget_score
  + 0.10 * time_score
  + 0.05 * group_score
  + 0.05 * quality_score
  + 0.05 * local_score
```

These are **hand-set starting weights**, not learned — see STEP/Section 16
below for how they should eventually be replaced.

---

## STEP 8 (spec Section 16) — ML strategy: what's real ML today vs future learning-to-rank

**Today, legitimately "ML":**
- Pretrained sentence embeddings (or the TF-IDF/SVD fallback) for semantic
  retrieval and similarity scoring — this is the actual ML component live in
  the system right now.
- Nothing here is fabricated as ML that isn't: the ranking *weights* are
  hand-set heuristics (explicitly labeled as "starting weights" throughout
  the code and this doc), not a trained model — Section 16's "do not
  fabricate training data" applies to weights too, not just labeled examples.

**Future: Learning-to-Rank, once real interaction data exists**

`feedback/feedback_store.py` and `POST /feedback` (Section 22) already
capture exactly the event stream Section 16 describes — `impression, click,
save, share, booking, visit, rating, reject, dwell_time` — tied to
`query_id` + `place_id`. Once enough of these accumulate:

1. **Label construction:** turn events into a relevance label per
   `(query, place)` pair — e.g. `impression`-only = 0, `click` = 1, `save`/
   `booking` = 2, `reject` = -1 (needs a real product decision, not asserted
   here).
2. **Feature vector:** the same features already computed per candidate today
   (`ScoreBreakdown`'s six components + raw place attributes + intent
   attributes) become the model's input features — nothing new to engineer,
   they already exist as a byproduct of today's heuristic scorer.
3. **Model:** a standard Learning-to-Rank model (e.g. LightGBM's `LambdaRank`
   objective, or XGBoost) trained on `(features) -> relevance label`,
   replacing the hand-set weights in `ranking/scoring.score_candidate` with
   a learned scoring function. The five-stage pipeline shape doesn't
   change — only Stage 4 swaps from a fixed formula to a model.
4. **Guardrail:** don't attempt this until there's a real, non-trivial volume
   of genuine interaction data — a model trained on a handful of hackathon
   demo clicks would overfit meaninglessly. Until then, the hand-set weights
   above are the correct, honest choice.

---

## Demo scripts (spec Section 28)

**Demo 1 — General Google engine:** `POST /recommend` (or `/recommend/google`)
with `"Find good cafes near Bangalore airport for 2 people."` → intent
extraction resolves city≠Mumbai → routes to the Google engine → (with a real
`GOOGLE_PLACES_API_KEY`) Text Search retrieves real cafes near the biased
location → semantic + hybrid ranking → top-K. Verified in `tests/test_service.py`
in mock mode (no key needed to demo the *pipeline shape*; add a key before
the actual judge demo for real places).

**Demo 2 — Mumbai local engine:** the spec's own Colaba example → intent
extraction resolves `city="Mumbai"` → routes to the Mumbai engine → retrieves
from the curated dataset → **Colaba Causeway Market ranks first**, exactly as
the spec predicted (verified in `tests/test_service.py::test_mumbai_demo_scenario_ranks_colaba_causeway_market_first`).
The Mumbai engine's richer fields (`experience_tags`, `traveller_tags`,
`local` block) are visible in each recommendation's `details`, giving the
judge-visible "richer than Google" contrast Section 28 asks for.

---

## Caching & performance (Sections 23–24)

| Data | Cached? | Where | Invalidation |
|---|---|---|---|
| Google Places query results | Yes, in-memory | `retrieval/google_places._query_cache` | TTL (`GOOGLE_PLACES_RESULT_CACHE_TTL_SECONDS`, default 1h) |
| Mumbai dataset (JSON) | Yes, in-memory | `retrieval/mumbai_retriever._load_json_dataset` (`lru_cache`) | Process restart (dataset is small/curated — edit-then-restart is fine for a hackathon) |
| Embeddings | Not cached on the hot path today; offline cache available | `data/embedding_cache.json` via `scripts/build_embeddings.py` | Manual re-run after dataset edits |
| Feedback | Append-only, never invalidated | `data/feedback_log.jsonl` or MySQL | N/A (write-once log) |

**What should be pre-computed vs real-time vs async, at hackathon scale:**
- Precomputed: nothing strictly required yet (dataset is small); the
  embedding cache script exists for when that changes.
- Real-time: intent extraction, retrieval, embedding, ranking — all fast
  enough (sub-second for the Mumbai engine; Google engine latency is
  dominated by the external API call, typically well under the "few
  seconds" target).
- Async: not needed at this scale; `POST /feedback` is a simple synchronous
  write and is intentionally best-effort (a failed feedback write never
  fails or blocks the recommendation request that preceded it).
