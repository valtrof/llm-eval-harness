import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRerankRetriever:
    """Two-stage retrieval: BM25 (sparse) and a dense bi-encoder each rank the full corpus,
    Reciprocal Rank Fusion merges those into one candidate pool, then a cross-encoder reranks
    that pool down to the final top_k by scoring each (query, doc) pair jointly."""

    BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CANDIDATE_POOL = 8
    RRF_K = 60

    def __init__(self, corpus_path: str = "data/corpus.json", top_k: int = 3):
        self.top_k = top_k
        with open(corpus_path) as f:
            self._corpus = json.load(f)
        texts = [doc["text"] for doc in self._corpus]

        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        self._bi_encoder = SentenceTransformer(self.BI_ENCODER_MODEL)
        self._cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
        self._corpus_embeddings = self._bi_encoder.encode(texts, normalize_embeddings=True)

    def _bm25_ranked(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        return list(np.argsort(-scores))

    def _dense_ranked(self, query: str) -> list[int]:
        query_embedding = self._bi_encoder.encode(query, normalize_embeddings=True)
        scores = self._corpus_embeddings @ query_embedding
        return list(np.argsort(-scores))

    def _rrf_fuse(self, ranked_lists: list[list[int]]) -> list[int]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked[: self.CANDIDATE_POOL], start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)
        return sorted(scores, key=lambda idx: -scores[idx])

    def retrieve(self, query: str) -> list[dict]:
        candidate_ids = self._rrf_fuse(
            [self._bm25_ranked(query), self._dense_ranked(query)]
        )[: self.CANDIDATE_POOL]

        pairs = [(query, self._corpus[i]["text"]) for i in candidate_ids]
        rerank_scores = self._cross_encoder.predict(pairs)

        reranked = sorted(zip(candidate_ids, rerank_scores), key=lambda x: -x[1])[: self.top_k]
        return [
            {**self._corpus[doc_id], "retrieval_score": float(score)}
            for doc_id, score in reranked
        ]
