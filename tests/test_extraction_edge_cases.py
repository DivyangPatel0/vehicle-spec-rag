"""Tests verifying real-world edge cases for vehicle specification extraction."""

import pytest
from src.schemas import DocumentChunk
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_store import VectorStore
from src.extraction.extractor import SpecificationExtractor
from src.extraction.llm_client import MockLLMClient, BaseLLMClient


class MalformedJSONClient(BaseLLMClient):
    """Simulates an LLM returning garbage or broken JSON."""
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return "Not valid JSON at all! {broken: true"


def test_edge_case_dual_units():
    """Edge Case: Extracts both metric (Nm) and imperial (lb-ft) properly."""
    context = "Brake Caliper Guide Pin Bolt: 35 Nm (26 lb-ft)"
    client = MockLLMClient()
    prompt = f"### Retrieved Service Manual Excerpts:\n{context}\n\n### User Query:\ncaliper guide pin bolt\n"
    res = client.generate(prompt)
    import json
    parsed = json.loads(res)
    assert len(parsed) == 1
    assert parsed[0]["value"] == "35"
    assert parsed[0]["unit"] == "Nm"
    assert parsed[0]["secondary_value"] == "26"
    assert parsed[0]["secondary_unit"] == "lb-ft"


def test_edge_case_multi_stage_angle_torque():
    """Edge Case: Multi-stage torques specifying angle/degree rotation."""
    context = "Cylinder head bolt: 40 Nm plus 90 degrees rotation."
    client = MockLLMClient()
    prompt = f"### Retrieved Service Manual Excerpts:\n{context}\n\n### User Query:\nCylinder head bolt torque\n"
    res = client.generate(prompt)
    import json
    parsed = json.loads(res)
    assert len(parsed) >= 1
    assert "90 deg" in parsed[0]["value"]
    assert parsed[0]["unit"] == "Nm"


def test_edge_case_tolerance_range():
    """Edge Case: Numerical ranges (e.g. 35-40 Nm)."""
    context = "Wheel hub bearing retaining nut: 35-40 Nm (26-30 lb-ft)."
    client = MockLLMClient()
    prompt = f"### Retrieved Service Manual Excerpts:\n{context}\n\n### User Query:\nWheel hub bearing retaining nut\n"
    res = client.generate(prompt)
    import json
    parsed = json.loads(res)
    assert len(parsed) == 1
    assert "35-40" in parsed[0]["value"] or "35" in parsed[0]["value"]


def test_edge_case_out_of_vocabulary_query():
    """Edge Case: Query asking for parts not mentioned in context must not hallucinate."""
    context = "Front Disc Brake Caliper Anchor Plate Bolt: 175 Nm (129 lb-ft)."
    client = MockLLMClient()
    prompt = f"### Retrieved Service Manual Excerpts:\n{context}\n\n### User Query:\nRear sunroof motor torque\n"
    res = client.generate(prompt)
    import json
    parsed = json.loads(res)
    assert parsed == []


def test_edge_case_malformed_llm_recovery():
    """Edge Case: Pipeline gracefully handles malformed LLM response without crashing."""
    chunk = DocumentChunk(
        chunk_id="c1",
        page_number=1,
        text="Brake Caliper Bolt: 35 Nm"
    )
    store = VectorStore()
    store.build_index([chunk])

    extractor = SpecificationExtractor(
        vector_store=store,
        llm_client=MalformedJSONClient()
    )
    result = extractor.extract("Brake Caliper Bolt")
    assert result.status == "parse_error"
    assert len(result.specifications) == 0
    assert "Failed to parse" in result.message
