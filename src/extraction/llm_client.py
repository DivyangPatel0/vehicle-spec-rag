"""Pluggable LLM clients supporting offline mock evaluation, OpenAI, and Gemini."""

import os
import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .prompts import EXTRACTION_SYSTEM_PROMPT


class BaseLLMClient(ABC):
    """Abstract interface for LLM extraction providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = EXTRACTION_SYSTEM_PROMPT) -> str:
        """Generates completion text from prompt."""
        pass


class MockLLMClient(BaseLLMClient):
    """
    High-fidelity deterministic offline extractor.
    Uses automotive domain regular expressions to extract structured specifications
    directly from context, enabling 100% offline testing, CI automation, and zero-cost evaluation.
    """

    # Matches: "Brake Caliper Bolt: 35 Nm (26 lb-ft)" or "Front wheel hub nuts - 200 Nm (148 lb-ft)"
    LINE_SPEC_PATTERN = re.compile(
        r"([A-Za-z0-9][A-Za-z0-9\s\-/,()]+?)\s*[:—\-–]\s*(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s*(Nm|N-m|N·m|lb-ft|lb-in|ft-lb|psi|kPa|bar|L|liters?|quarts|mm)\b(?:\s*[\(\[]\s*(\d+(?:\.\d+)?)\s*(lb-ft|lb-in|ft-lb|in-lb|Nm|psi)\s*[\)\]])?",
        re.IGNORECASE
    )

    # Matches multi-stage torque specs: "40 Nm plus 90 degrees" or "Stage 1: 30 Nm, Stage 2: 90 deg"
    STAGE_TORQUE_PATTERN = re.compile(
        r"([A-Za-z0-9][A-Za-z0-9\s\-/,()]+?)\s*[:—\-–]\s*(\d+)\s*(Nm)\s*(?:plus|\+)\s*(\d+)\s*(?:deg|degrees|°)",
        re.IGNORECASE
    )

    # Matches table row: "Caliper anchor plate bolts   175   129" or split across lines:
    TABLE_ROW_PATTERN = re.compile(
        r"([A-Za-z][A-Za-z0-9\s\-/,()]{3,50}?)(?:\s{2,}|\s*\n\s*)(\d+(?:\.\d+)?)(?:\s{2,}|\s+|\n\s*)(\d+(?:\.\d+)?)",
        re.MULTILINE
    )

    def generate(self, prompt: str, system_prompt: str = EXTRACTION_SYSTEM_PROMPT) -> str:
        # Extract query and context from the prompt
        query_match = re.search(r"### User Query:\s*(.*?)(?=\n\n|\nExtract|\Z)", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ""

        context_match = re.search(r"### Retrieved Service Manual Excerpts:\s*(.*?)(?=### User Query:|\Z)", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else prompt

        # Query keywords for relevance filtering
        generic_stops = {
            "torque", "spec", "specs", "specification", "specifications", "value", "values",
            "the", "for", "and", "nut", "nuts", "bolt", "bolts", "screw", "screws", "fastener", "fasteners"
        }
        all_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        core_query_terms = all_words - generic_stops
        query_terms = core_query_terms if core_query_terms else all_words

        extracted_items: List[Dict[str, Any]] = []
        seen_keys = set()

        # 1. Check for stage torques
        for m in self.STAGE_TORQUE_PATTERN.finditer(context):
            comp, val1, u1, deg = m.groups()
            comp_clean = comp.strip().strip("-•* \t\n")
            if any(term in comp_clean.lower() for term in query_terms) or not query_terms:
                key = (comp_clean.lower(), f"{val1} Nm + {deg} deg")
                if key not in seen_keys:
                    seen_keys.add(key)
                    extracted_items.append({
                        "component": comp_clean.title(),
                        "spec_type": "Torque",
                        "value": f"{val1} + {deg} deg",
                        "unit": u1,
                        "secondary_value": None,
                        "secondary_unit": None,
                        "notes": f"Angle torque tightened to {val1} {u1} plus {deg} degrees rotation"
                    })

        # 2. Check standard line specifications
        for m in self.LINE_SPEC_PATTERN.finditer(context):
            comp, val, unit, sec_val, sec_unit = m.groups()
            comp_clean = comp.strip().strip("-•* \t\n")
            if len(comp_clean) < 3 or len(comp_clean) > 80:
                continue

            # Determine spec type
            unit_norm = unit.strip()
            if unit_norm.lower() in ["nm", "n-m", "n·m", "lb-ft", "lb-in", "ft-lb"]:
                spec_type = "Torque"
                unit_norm = "Nm" if unit_norm.lower().startswith("n") else "lb-ft"
            elif unit_norm.lower() in ["l", "liter", "liters", "quarts"]:
                spec_type = "Fluid Capacity"
            elif unit_norm.lower() in ["psi", "kpa", "bar"]:
                spec_type = "Pressure"
            else:
                spec_type = "Clearance"

            # Filter by query terms if any terms were found in query
            if query_terms and not any(term in comp_clean.lower() for term in query_terms):
                continue

            key = (comp_clean.lower(), val, unit_norm)
            if key not in seen_keys:
                seen_keys.add(key)
                extracted_items.append({
                    "component": comp_clean.title(),
                    "spec_type": spec_type,
                    "value": str(val).strip(),
                    "unit": unit_norm,
                    "secondary_value": str(sec_val).strip() if sec_val else None,
                    "secondary_unit": str(sec_unit).strip() if sec_unit else None,
                    "notes": None
                })

        # 3. Check table rows if in a torque or spec table section
        if any(w in context.lower() for w in ["torque", "spec", "description", "nm", "lb-ft", "lb-in"]):
            for m in self.TABLE_ROW_PATTERN.finditer(context):
                comp, val1, val2 = m.groups()
                comp_lines = [l.strip() for l in comp.splitlines() if l.strip()]
                filtered_lines = [l for l in comp_lines if not any(w in l.lower() for w in ["nm", "lb-ft", "lb-in", "description", "specification"])]
                comp_clean = (filtered_lines[-1] if filtered_lines else comp_lines[-1]).strip().strip("-•* \t\n")
                if len(comp_clean) < 3 or comp_clean.lower().startswith(("item", "page", "section")):
                    continue
                if query_terms and not any(term in comp_clean.lower() for term in query_terms):
                    continue

                key = (comp_clean.lower(), val1, "Nm")
                if key not in seen_keys:
                    seen_keys.add(key)
                    extracted_items.append({
                        "component": comp_clean.title(),
                        "spec_type": "Torque",
                        "value": str(val1).strip(),
                        "unit": "Nm",
                        "secondary_value": str(val2).strip(),
                        "secondary_unit": "lb-ft",
                        "notes": None
                    })

        return json.dumps(extracted_items, indent=2)


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client using structured JSON generation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run 'pip install openai'.")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str = EXTRACTION_SYSTEM_PROMPT) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content or "[]"


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """Factory creating configured LLM client."""
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAILLMClient()
    return MockLLMClient()
