"""Tests for semantic and table-aware chunking."""

import pytest
from src.chunking.chunker import SemanticChunker

def test_chunker_basic_splitting():
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    page_data = {
        "page_number": 42,
        "text": "Paragraph one with some text.\n\nParagraph two with another text.\n\nParagraph three.",
        "section_title": "Brakes"
    }
    chunks = chunker.chunk_page(page_data)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.page_number == 42
        assert c.section_title == "Brakes"
        assert c.chunk_id.startswith("p42_c")


def test_chunker_empty_page():
    chunker = SemanticChunker()
    chunks = chunker.chunk_page({"page_number": 1, "text": "   "})
    assert chunks == []


def test_chunker_detects_torque_spec_units():
    chunker = SemanticChunker(chunk_size=300)
    text = (
        "TORQUE SPECIFICATIONS\n\n"
        "Brake caliper guide pin bolt: 35 Nm (26 lb-ft)\n"
        "Anchor plate bolts: 175 Nm (129 lb-ft)"
    )
    chunks = chunker.chunk_page({"page_number": 10, "text": text})
    assert len(chunks) > 0
    assert chunks[0].is_table_row is True
