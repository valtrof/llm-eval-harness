import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, corpus_path: str = "data/corpus.json", top_k: int = 3):
        self.top_k = top_k
        with open(corpus_path) as f:
            self._corpus = json.load(f)
        texts = [doc["text"] for doc in self._corpus]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str) -> list[dict]:
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [
            {**self._corpus[i], "retrieval_score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]
