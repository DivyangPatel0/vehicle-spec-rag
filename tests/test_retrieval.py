"""Tests for embedding generation and vector store retrieval."""

import pytest
import numpy as np
from src.schemas import DocumentChunk
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_store import VectorStore


def test_embedder_and_search(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            page_number=10,
            text="Front disc brake caliper anchor plate bolt torque is 175 Nm.",
            section_title="Brakes"
        ),
        DocumentChunk(
            chunk_id="c2",
            page_number=15,
            text="Engine oil pan drain plug torque is 25 Nm. Engine oil capacity is 5.7 Liters.",
            section_title="Engine Lubrication"
        ),
        DocumentChunk(
            chunk_id="c3",
            page_number=20,
            text="Front suspension stabilizer bar link nut torque is 80 Nm.",
            section_title="Front Suspension"
        )
    ]

    embedder = TextEmbedder()
    store = VectorStore(embedder=embedder)
    store.build_index(chunks)

    # Search for brake caliper
    results = store.search("brake caliper bolt torque", top_k=2)
    assert len(results) >= 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert "caliper" in top_chunk.text.lower()
    assert score > 0.1

    # Test serialization
    cache_path = tmp_path / "test_index.pkl"
    store.save(cache_path)
    assert cache_path.exists()

    # Load and verify
    loaded_store = VectorStore.load(cache_path)
    loaded_results = loaded_store.search("brake caliper", top_k=1)
    assert len(loaded_results) == 1
    assert loaded_results[0][0].chunk_id == "c1"
