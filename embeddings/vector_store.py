"""
Vector similarity search over a per-request candidate set.

For a hackathon-scale system this is intentionally a brute-force numpy
cosine-similarity search, NOT FAISS/Qdrant/Pinecone, because:

  - Candidate sets are already small by design (20-100 places — see
    retrieval/candidate_retriever.py and Section 8): brute force over 100
    384-dim vectors is sub-millisecond. A vector DB adds infra + ops cost
    with zero latency benefit at this scale.
  - The Mumbai dataset itself starts at a few dozen to a few hundred rows
    (Section 10) — small enough to embed in full and re-rank per query
    without an index.

WHEN TO GRADUATE to FAISS or pgvector (see DESIGN.md Step 6 for the full
comparison table): once the Mumbai dataset grows past roughly 5,000-10,000
providers, or you want to pre-filter the Mumbai dataset semantically BEFORE
the geo/category candidate filter (i.e. semantic search IS the candidate
retrieval, not just the re-ranking step). At that point, swap this module's
internals for a FAISS `IndexFlatIP` (exact) or `IndexIVFFlat` (approximate);
the public `search()` interface below would not need to change.
"""
from __future__ import annotations

import numpy as np


class InMemoryVectorStore:
    """Holds one query's candidate embeddings; not persisted between requests."""

    def __init__(self, ids: list[str], vectors: np.ndarray):
        assert len(ids) == vectors.shape[0]
        self.ids = ids
        # Vectors are expected pre-normalized (Embedder does this); normalize
        # defensively anyway so cosine similarity == dot product below.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vectors / norms

    def search(self, query_vector: np.ndarray, top_k: int | None = None) -> list[tuple[str, float]]:
        """Returns [(id, cosine_similarity), ...] sorted descending."""
        if self.vectors.shape[0] == 0:
            return []
        qnorm = np.linalg.norm(query_vector)
        q = query_vector / qnorm if qnorm > 0 else query_vector
        sims = self.vectors @ q
        order = np.argsort(-sims)
        if top_k is not None:
            order = order[:top_k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def similarity_map(self, query_vector: np.ndarray) -> dict[str, float]:
        return dict(self.search(query_vector, top_k=None))
