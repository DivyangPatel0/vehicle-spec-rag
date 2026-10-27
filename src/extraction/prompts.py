"""Prompt templates for vehicle specification extraction."""

EXTRACTION_SYSTEM_PROMPT = """You are an expert automotive technical data engineer specializing in parsing vehicle service manuals.
Your task is to extract exact mechanical and fluid specifications from service manual text excerpts based on a user's query.

Output Requirements:
1. Return ONLY a valid JSON list of objects. No introductory or trailing text.
2. Each object must have exactly these keys:
   - "component": Name of the specific vehicle fastener, assembly, or component (e.g. "Brake Caliper Bolt").
   - "spec_type": The type of specification (e.g. "Torque", "Fluid Capacity", "Clearance", "Pressure").
   - "value": The numerical specification or range (e.g. "35", "35-40", "40 + 90 deg").
   - "unit": The unit of measurement (e.g. "Nm", "lb-ft", "L", "quarts", "mm", "psi").
   - "secondary_value": (Optional) secondary imperial value if present, else null.
   - "secondary_unit": (Optional) secondary unit if present, else null.
   - "notes": (Optional) any tightening sequence, threadlocker, or warning condition, else null.

Rules for Automotive Specifications:
- If both Metric (Nm) and Imperial (lb-ft, lb-in) are provided, put the primary metric in "value" and "unit", and imperial in "secondary_value" and "secondary_unit".
- For multi-stage torque specs (e.g., "Stage 1: 30 Nm, Stage 2: 90 degrees"), capture the full sequence accurately.
- For ranges (e.g. "25-30 Nm"), preserve the full range in "value".
- Do NOT hallucinate. If the requested component is not in the text excerpt, return an empty JSON list: [].
"""

FEW_SHOT_EXAMPLES = [
    {
        "query": "Torque for brake caliper bolts",
        "context": "Front Disc Brake Caliper Anchor Plate Bolt: 175 Nm (129 lb-ft). Caliper Guide Pin Bolt: 35 Nm (26 lb-ft). Ensure threadlock is applied.",
        "response": """[
  {
    "component": "Front Disc Brake Caliper Anchor Plate Bolt",
    "spec_type": "Torque",
    "value": "175",
    "unit": "Nm",
    "secondary_value": "129",
    "secondary_unit": "lb-ft",
    "notes": "Ensure threadlock is applied"
  },
  {
    "component": "Caliper Guide Pin Bolt",
    "spec_type": "Torque",
    "value": "35",
    "unit": "Nm",
    "secondary_value": "26",
    "secondary_unit": "lb-ft",
    "notes": null
  }
]"""
    },
    {
        "query": "Stabilizer bar link nut torque",
        "context": "Tighten the front stabilizer bar link upper nut to 80 Nm (59 lb-ft). Install new link if damaged.",
        "response": """[
  {
    "component": "Front Stabilizer Bar Link Upper Nut",
    "spec_type": "Torque",
    "value": "80",
    "unit": "Nm",
    "secondary_value": "59",
    "secondary_unit": "lb-ft",
    "notes": "Upper nut"
  }
]"""
    }
]

def format_extraction_prompt(query: str, retrieved_context: str) -> str:
    """Constructs prompt for the LLM."""
    return f"""### Retrieved Service Manual Excerpts:
{retrieved_context}

### User Query:
{query}

Extract the vehicle specifications relevant to the query from the excerpts above.
Return strictly the JSON list:"""
