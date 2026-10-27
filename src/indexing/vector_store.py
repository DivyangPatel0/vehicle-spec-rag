"""In-memory persistent vector store with cosine similarity retrieval."""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from ..schemas import DocumentChunk
from .embedder import TextEmbedder


class VectorStore:
    """Stores chunks and embeddings with persistent serialization and cosine search."""

    def __init__(self, embedder: Optional[TextEmbedder] = None):
        self.embedder = embedder or TextEmbedder()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, chunks: List[DocumentChunk]):
        """Builds vector index from a list of DocumentChunk objects."""
        if not chunks:
            raise ValueError("Cannot build index with empty chunks list")
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.embedder.embed_documents(texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top-k most relevant chunks for a query.
        Returns a list of (DocumentChunk, score) tuples.
        """
        if not self.chunks or self.embeddings is None:
            return []

        query_vec = self.embedder.embed_query(query)
        # Cosine similarity is the dot product of L2-normalized vectors
        scores = np.dot(self.embeddings, query_vec)

        # Sort indices by descending score
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < threshold:
                break
            results.append((self.chunks[idx], score))
            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: str | Path):
        """Serializes the index and chunks to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "embedder": self.embedder
            }, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "VectorStore":
        """Loads a serialized index from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found at: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls(embedder=data.get("embedder"))
        store.chunks = data["chunks"]
        store.embeddings = data["embeddings"]
        return store
