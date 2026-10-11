"""
Embedding generation (Section 13).

Two backends behind one interface (`Embedder.encode`):

  1. "st" — Sentence-Transformers, `all-MiniLM-L6-v2` (384 dims). This is the
     RECOMMENDED backend for real use: small (~80MB), fast on CPU (fine for
     a hackathon demo, no GPU needed), and a strong general-purpose semantic
     similarity model. Requires `pip install sentence-transformers` and a
     one-time model download (needs internet the first time; cached locally
     after that — see DESIGN.md for offline-caching instructions).

  2. "hashing" — a dependency-light TF-IDF + SVD fallback (scikit-learn),
     used automatically when sentence-transformers isn't installed or the
     model can't be downloaded (e.g. no internet in a sandboxed CI/dev
     environment). It is NOT a substitute for real semantic embeddings —
     it won't catch synonyms it hasn't seen in the corpus — but it keeps
     the pipeline, tests, and demo runnable with zero setup, and gives a
     concrete fallback contract.

Select with config.EMBEDDING_BACKEND = "auto" (default) | "st" | "hashing".
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from config import settings

logger = logging.getLogger("travelsathi.embeddings")

_lock = threading.Lock()
_singleton: "Embedder | None" = None


class Embedder:
    def __init__(self, backend: str = "auto"):
        self.backend_name = backend
        self._st_model = None
        self._hashing_vectorizer = None
        self._svd = None
        self._fitted = False

        if backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                self.backend_name = "st"
                self.dimensions = self._st_model.get_sentence_embedding_dimension()
                logger.info("Embedder using sentence-transformers (%s, dim=%d).",
                            settings.EMBEDDING_MODEL_NAME, self.dimensions)
                return
            except Exception as exc:
                if backend == "st":
                    raise
                logger.warning("sentence-transformers unavailable (%s); using hashing fallback.", exc)

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._hashing_vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        self.backend_name = "hashing"
        self.dimensions = 256  # after SVD reduction
        logger.info("Embedder using offline TF-IDF/SVD fallback (dim=%d).", self.dimensions)

    def _fit_hashing(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD

        matrix = self._hashing_vectorizer.fit_transform(corpus)
        n_components = min(self.dimensions, max(2, matrix.shape[1] - 1), max(2, matrix.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(matrix)
        self._fitted = True

    def fit(self, corpus: list[str]) -> None:
        """Only meaningful for the hashing backend — no-op for sentence-transformers."""
        if self.backend_name == "hashing" and corpus:
            self._fit_hashing(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        if self.backend_name == "st":
            vectors = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)

        # hashing backend
        if not self._fitted:
            self._fit_hashing(texts)
        tfidf = self._hashing_vectorizer.transform(texts)
        reduced = self._svd.transform(tfidf)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (reduced / norms).astype(np.float32)


def get_embedder() -> Embedder:
    """Process-wide singleton so we don't reload the model per-request."""
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = Embedder(settings.EMBEDDING_BACKEND)
        return _singleton
