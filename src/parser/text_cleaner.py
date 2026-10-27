"""Text cleaning and normalization utilities for automotive service manual PDFs."""

import re
import unicodedata

# Unicode ligatures mapping to standard ASCII equivalents
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "•",
    "\u00a0": " ",  # Non-breaking space
}

# Regex to detect recurring service manual headers / footers
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^200\d-\d{2}\s+[A-Za-z\s]+.*$", re.MULTILINE),
    re.compile(r"^Section\s+\d{3}-\d{2}:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.MULTILINE | re.IGNORECASE),
]

def normalize_ligatures(text: str) -> str:
    """Replaces Unicode typographical ligatures with standard characters."""
    if not text:
        return ""
    for ligature, replacement in LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return unicodedata.normalize("NFKC", text)

def clean_extracted_text(text: str, remove_headers: bool = True) -> str:
    """Cleans raw PDF text while preserving technical specifications and numbers."""
    if not text:
        return ""

    # 1. Normalize ligatures & Unicode
    text = normalize_ligatures(text)

    # 2. Remove repetitive service manual running headers/footers if requested
    if remove_headers:
        for pattern in HEADER_FOOTER_PATTERNS:
            text = pattern.sub("", text)

    # 3. Normalize bullet points and list indicators
    text = re.sub(r"^[z•\*\-]\s*", "- ", text, flags=re.MULTILINE)

    # 4. Normalize whitespace while preserving line breaks for tables and lists
    lines = []
    for line in text.splitlines():
        line_stripped = " ".join(line.split())
        if line_stripped:
            lines.append(line_stripped)

    # Join cleaned lines
    cleaned = "\n".join(lines)
    return cleaned
