"""Pydantic data schemas for vehicle specifications, chunks, and extraction results."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SpecificationItem(BaseModel):
    """Structured vehicle specification as required by the assignment specification."""
    component: str = Field(
        ...,
        description="Name of the vehicle part, fastener, or component (e.g. Brake Caliper Bolt)"
    )
    spec_type: str = Field(
        default="Torque",
        description="Type of specification (e.g., Torque, Capacity, Clearance, Pressure, Gap)"
    )
    value: str = Field(
        ...,
        description="Specification value or range (e.g., '35', '35-40', '40 + 90 deg')"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g., 'Nm', 'lb-ft', 'psi', 'L', 'quarts', 'mm')"
    )
    secondary_value: Optional[str] = Field(
        default=None,
        description="Secondary/imperial measurement value if dual units exist (e.g., '26')"
    )
    secondary_unit: Optional[str] = Field(
        default=None,
        description="Secondary/imperial unit (e.g., 'lb-ft')"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Special conditions (e.g. 'Use new bolts', 'Stage 1 tightening')"
    )
    source_page: Optional[int] = Field(
        default=None,
        description="Service manual page number where this specification was found"
    )
    confidence: Optional[float] = Field(
        default=1.0,
        description="Extraction confidence score between 0.0 and 1.0"
    )

    @field_validator("component")
    @classmethod
    def clean_component(cls, v: str) -> str:
        cleaned = v.strip().strip("-•* \t\n")
        if not cleaned:
            raise ValueError("Component name cannot be empty")
        return cleaned

    @field_validator("value")
    @classmethod
    def clean_value(cls, v: str) -> str:
        cleaned = str(v).strip()
        if not cleaned:
            raise ValueError("Specification value cannot be empty")
        return cleaned

    @field_validator("unit")
    @classmethod
    def clean_unit(cls, v: str) -> str:
        cleaned = str(v).strip()
        return cleaned or "N/A"

    def to_submission_dict(self) -> Dict[str, Any]:
        """Returns standard dict conforming strictly to assignment format."""
        return {
            "component": self.component,
            "spec_type": self.spec_type,
            "value": self.value,
            "unit": self.unit
        }


class ExtractionResult(BaseModel):
    """Container for the output of a retrieval-augmented query."""
    query: str
    specifications: List[SpecificationItem] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    pages_referenced: List[int] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    status: str = "success"
    message: Optional[str] = None

    def to_clean_list(self) -> List[Dict[str, Any]]:
        """Converts results to assignment-specified JSON list format."""
        return [item.to_submission_dict() for item in self.specifications]


class DocumentChunk(BaseModel):
    """Text chunk with provenance metadata."""
    chunk_id: str
    page_number: int
    text: str
    section_title: Optional[str] = None
    char_count: int = 0
    token_estimate: int = 0
    is_table_row: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self.char_count = len(self.text)
        self.token_estimate = max(1, len(self.text) // 4)
