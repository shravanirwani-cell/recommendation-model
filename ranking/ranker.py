"""
Orchestrates Stages 2-5 of the pipeline (Section 7):

    candidates (NormalizedPlace list)
        -> Stage 2: hard constraint filtering
        -> Stage 3: semantic relevance (embeddings + cosine similarity)
        -> Stage 4: hybrid ranking (semantic + structured features)
        -> Stage 5: top-K

Used identically by both engines — this is "the one recommendation
algorithm" from Section 17, operating purely on NormalizedPlace, with no
idea whether a candidate came from Google or the Mumbai dataset.
"""
from __future__ import annotations

import logging

from embeddings.embedder import get_embedder
from embeddings.vector_store import InMemoryVectorStore
from models.schemas import EngineMeta, Intent, NormalizedPlace, Recommendation
from normalization.normalized_place import build_embedding_text
from ranking import feature_engineering as fe
from ranking.scoring import build_match_reasons, check_hard_constraints, score_candidate

logger = logging.getLogger("travelsathi.ranking")


def _opening_status(place: NormalizedPlace) -> str:
    if place.opening_hours.open_now is True:
        return "open"
    if place.opening_hours.open_now is False:
        return "closed"
    return "unknown"


def rank_candidates(
    intent: Intent,
    candidates: list[NormalizedPlace],
    engine: str,
    top_k: int = 10,
) -> tuple[list[Recommendation], EngineMeta]:
    warnings: list[str] = []
    total_considered = len(candidates)

    # --- Stage 2: hard constraint filtering ---
    survivors: list[NormalizedPlace] = []
    for place in candidates:
        reason = check_hard_constraints(intent, place)
        if reason is None:
            survivors.append(place)
        else:
            logger.debug("Filtered %s: %s", place.id, reason)

    if not survivors:
        warnings.append("All candidates were excluded by hard constraints; returning empty result set.")
        return [], EngineMeta(
            candidates_considered=total_considered,
            candidates_after_constraints=0,
            engine=engine,
            warnings=warnings,
        )

    # --- Stage 3: semantic relevance ---
    embedder = get_embedder()
    for place in survivors:
        if not place.embedding_text:
            place.embedding_text = build_embedding_text(place)

    query_text = intent.raw_query
    corpus = [p.embedding_text for p in survivors] + [query_text]
    embedder.fit(corpus)  # no-op for the sentence-transformers backend

    candidate_vectors = embedder.encode([p.embedding_text for p in survivors])
    query_vector = embedder.encode([query_text])[0]

    store = InMemoryVectorStore([p.id for p in survivors], candidate_vectors)
    similarity_map = store.similarity_map(query_vector)
    # Cosine similarity is in [-1, 1]; rescale to [0, 1] for blending with
    # the other [0, 1] features.
    semantic_scores = {pid: (sim + 1) / 2 for pid, sim in similarity_map.items()}

    # --- Stage 4: hybrid ranking ---
    recommendations: list[Recommendation] = []
    for place in survivors:
        sem_score = semantic_scores.get(place.id, 0.5)
        breakdown, constraint_score = score_candidate(intent, place, sem_score)
        reasons = build_match_reasons(intent, place, breakdown)

        distance = fe.distance_km(intent, place)
        cost_val = place.cost.max_inr if place.cost.max_inr is not None else place.cost.min_inr

        recommendations.append(Recommendation(
            place_id=place.id,
            source=place.source,
            name=place.name,
            address=place.address,
            latitude=place.latitude,
            longitude=place.longitude,
            category=place.categories,
            distance_km=round(distance, 2) if distance is not None else None,
            estimated_cost_per_person=cost_val,
            cost_note=place.cost.note,
            rating=place.rating,
            opening_status=_opening_status(place),
            semantic_score=breakdown.semantic_score,
            constraint_score=constraint_score,
            final_score=breakdown.final_score,
            match_reasons=reasons,
            score_breakdown=breakdown,
            details=place.model_dump(),
        ))

    # --- Stage 5: top-K ---
    recommendations.sort(key=lambda r: r.final_score, reverse=True)
    top = recommendations[:top_k]

    logger.info(
        "Ranked %d/%d candidates for engine=%s; returning top %d.",
        len(survivors), total_considered, engine, len(top),
    )

    return top, EngineMeta(
        candidates_considered=total_considered,
        candidates_after_constraints=len(survivors),
        engine=engine,
        data_source=("json/mysql" if engine == "mumbai" else "google_places_api/mock"),
        warnings=warnings,
    )
