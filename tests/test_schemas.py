"""Tests for Pydantic schema validation and format conformance."""

import pytest
from src.schemas import SpecificationItem, ExtractionResult


def test_specification_item_valid():
    item = SpecificationItem(
        component="Brake Caliper Bolt",
        spec_type="Torque",
        value="35",
        unit="Nm",
        secondary_value="26",
        secondary_unit="lb-ft"
    )
    d = item.to_submission_dict()
    assert d == {
        "component": "Brake Caliper Bolt",
        "spec_type": "Torque",
        "value": "35",
        "unit": "Nm"
    }


def test_specification_item_empty_component_validation():
    with pytest.raises(ValueError):
        SpecificationItem(component="   ", value="35", unit="Nm")


def test_specification_item_empty_value_validation():
    with pytest.raises(ValueError):
        SpecificationItem(component="Caliper Bolt", value="   ", unit="Nm")


def test_extraction_result_clean_list():
    res = ExtractionResult(
        query="Torque for brake caliper bolts",
        specifications=[
            SpecificationItem(component="Brake Caliper Bolt", spec_type="Torque", value="35", unit="Nm")
        ],
        retrieved_chunks_count=1,
        pages_referenced=[12]
    )
    clean_list = res.to_clean_list()
    assert len(clean_list) == 1
    assert clean_list[0]["component"] == "Brake Caliper Bolt"
    assert "retrieved_chunks_count" not in clean_list[0]
