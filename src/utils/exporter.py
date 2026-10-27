"""Export functions for serializing vehicle specifications to JSON and CSV."""

import json
from pathlib import Path
from typing import List, Union
import pandas as pd

from ..schemas import SpecificationItem, ExtractionResult


def export_to_json(
    items: Union[ExtractionResult, List[SpecificationItem]],
    filepath: str | Path,
    strict_assignment_format: bool = True
) -> Path:
    """
    Exports specification items to a JSON file.
    If strict_assignment_format is True, outputs strictly:
    [
      { "component": "...", "spec_type": "...", "value": "...", "unit": "..." }
    ]
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(items, ExtractionResult):
        specs = items.specifications
    else:
        specs = items

    if strict_assignment_format:
        data = [s.to_submission_dict() for s in specs]
    else:
        data = [s.model_dump() for s in specs]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def export_to_csv(
    items: Union[ExtractionResult, List[SpecificationItem]],
    filepath: str | Path,
    strict_assignment_format: bool = True
) -> Path:
    """
    Exports specification items to a CSV file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(items, ExtractionResult):
        specs = items.specifications
    else:
        specs = items

    if strict_assignment_format:
        data = [s.to_submission_dict() for s in specs]
    else:
        data = [s.model_dump() for s in specs]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8")
    return path
