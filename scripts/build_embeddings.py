"""
Offline batch embedding generation for the Mumbai dataset (Section 13).

Run manually whenever the Mumbai dataset changes:

    python scripts/build_embeddings.py

Writes data/embedding_cache.json: {provider_id: {"text": ..., "vector": [...]}}.

WHY THIS IS SEPARATE FROM THE REQUEST PATH:
Today's dataset (a few dozen rows) is cheap enough to re-embed on every
request (see ranking/ranker.py) — that's why the live pipeline does NOT
read this cache yet. Once the Mumbai dataset grows into the hundreds/
thousands, re-embedding on every request becomes wasteful CPU, and this
script's cache is what ranker.py should switch to reading (embed the query
only at request time; look up candidate vectors from this cache instead of
calling embedder.encode() on them). That's a small, isolated change:
retrieval/mumbai_retriever.py would attach cached vectors to each
NormalizedPlace instead of leaving embedding_text to be embedded fresh.

UPDATING: re-run this script after any edit to data/mumbai_places.json (or
after syncing new rows from MySQL). It's idempotent — safe to re-run anytime.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from embeddings.embedder import get_embedder  # noqa: E402
from normalization.normalized_place import build_embedding_text, from_mumbai_record  # noqa: E402


def main() -> None:
    with open(settings.MUMBAI_DATA_PATH, "r", encoding="utf-8") as f:
        raw_records = json.load(f).get("places", [])

    places = [from_mumbai_record(r) for r in raw_records]
    texts = [build_embedding_text(p) for p in places]

    embedder = get_embedder()
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    cache = {
        place.source_id: {"text": text, "vector": vector.tolist()}
        for place, text, vector in zip(places, texts, vectors)
    }

    os.makedirs(os.path.dirname(settings.EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(settings.EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"backend": embedder.backend_name, "dimensions": embedder.dimensions, "places": cache}, f, indent=2)

    print(f"Wrote {len(cache)} embeddings ({embedder.backend_name}, dim={embedder.dimensions}) "
          f"to {settings.EMBEDDING_CACHE_PATH}")


if __name__ == "__main__":
    main()
