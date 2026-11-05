"""
kNN retrieval in sentence-embedding space (Wang et al., App. C.1).

Embeds the question stem with paraphrase-MiniLM-L6-v2, L2-normalised, so a dot
product is cosine similarity. Retrieved demonstrations are ordered from least to
most similar, so the closest example sits right before the query. The paper does
not state an order; this choice is fixed across every arm.
"""

import zlib

import numpy as np


class Retriever:
    def __init__(self, embedder_name: str):
        self.name = embedder_name
        self._model = None
        self._cache = {}

    def _encode(self, texts):
        if self.name == "fake":
            # Deterministic pseudo-embeddings for offline tests.
            out = []
            for t in texts:
                rng = np.random.default_rng(zlib.crc32(t.encode()))
                v = rng.normal(size=32)
                out.append(v / np.linalg.norm(v))
            return np.array(out)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                  batch_size=256)

    def embed(self, items):
        keys = [it.question for it in items]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if missing:
            for k, v in zip(missing, self._encode(missing)):
                self._cache[k] = v
        return np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 1))

    def knn(self, query, pool, n):
        """Top-n most similar pool items, ordered least -> most similar."""
        if not pool:
            return []
        if len(pool) <= n:
            idx = np.argsort(self.embed(pool) @ self.embed([query])[0])
            return [pool[i] for i in idx]
        scores = self.embed(pool) @ self.embed([query])[0]
        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(scores[top])]
        return [pool[i] for i in top]

    def filter_pool(self, pool, queries, c):
        """Algorithm 2: keep, per server query, its c nearest pool items (union)."""
        if len(pool) <= c:
            return list(pool)
        P = self.embed(pool)
        Q = self.embed(queries)
        keep = set()
        for qv in Q:
            s = P @ qv
            keep.update(int(i) for i in np.argpartition(s, -c)[-c:])
        return [pool[i] for i in sorted(keep)]
