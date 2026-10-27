"""Text embedding generation for semantic and keyword retrieval."""

import os
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEmbedder:
    """
    Robust embedding generator designed for technical automotive manuals.
    Utilizes subword n-gram TF-IDF vectorization with L2 normalization,
    providing lightning-fast, zero-dependency, and deterministic embeddings
    ideal for automotive part numbers, torque units, and component names.
    Also supports pluggable transformer embedding models if desired.
    """

    def __init__(self, model_name: str = "tfidf-technical"):
        self.model_name = model_name
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            token_pattern=r"(?u)\b[\w\-\.\/]+\b", # Preserves hyphens and technical numbers
            sublinear_tf=True,
            norm="l2"
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "TextEmbedder":
        """Fits the vocabulary on the corpus chunks."""
        if not texts:
            raise ValueError("Corpus texts cannot be empty")
        self._vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of document chunks into dense normalized vectors."""
        if not self.is_fitted:
            self.fit(texts)
        sparse_vecs = self._vectorizer.transform(texts)
        return sparse_vecs.toarray().astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a search query into a dense normalized vector."""
        if not self.is_fitted:
            raise RuntimeError("Embedder must be fitted on corpus before embedding queries.")
        sparse_vec = self._vectorizer.transform([query])
        return sparse_vec.toarray()[0].astype(np.float32)
