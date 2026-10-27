"""Tests for PDF loading and text cleaning."""

import pytest
from pathlib import Path
from src.parser.text_cleaner import clean_extracted_text, normalize_ligatures
from src.parser.pdf_loader import PDFLoader

SAMPLE_PDF = Path("sample-service-manual 1.pdf")


def test_ligature_normalization():
    """Verifies Unicode typographical ligatures are converted to standard ASCII."""
    raw = "Speci\ufb01cation of the \ufb02uid \ufb00low, di\ufb00erential, and ba\ufb04e"
    cleaned = normalize_ligatures(raw)
    assert "Specification" in cleaned
    assert "fluid" in cleaned
    assert "flow" in cleaned
    assert "differential" in cleaned
    assert "baffle" in cleaned


def test_text_cleaner_headers_and_bullets():
    """Verifies recurring headers are stripped and bullets normalized."""
    sample = (
        "2004-06 F-150 Workshop Manual\n"
        "Section 204-00: Suspension System\n"
        "z Road test.\n"
        "• Inspect tires.\n"
        "Page 1 of 852"
    )
    cleaned = clean_extracted_text(sample)
    assert "- Road test." in cleaned
    assert "- Inspect tires." in cleaned
    assert "Page 1 of 852" not in cleaned


def test_pdf_loader_invalid_file():
    """Verifies FileNotFoundError is raised on non-existent path."""
    with pytest.raises(FileNotFoundError):
        PDFLoader("non_existent_file.pdf")


def test_pdf_loader_sample_manual():
    """Verifies reading sample service manual if file exists."""
    if not SAMPLE_PDF.exists():
        pytest.skip("sample-service-manual 1.pdf not present in test root")

    with PDFLoader(SAMPLE_PDF) as loader:
        assert loader.total_pages > 800
        page1 = loader.extract_page_text(1)
        assert page1["page_number"] == 1
        assert "Suspension System" in page1["text"]
        assert len(page1["text"]) > 100


def test_pdf_loader_out_of_bounds():
    """Verifies exception when requesting out-of-bounds page."""
    if not SAMPLE_PDF.exists():
        pytest.skip("sample-service-manual 1.pdf not present in test root")

    with PDFLoader(SAMPLE_PDF) as loader:
        with pytest.raises(IndexError):
            loader.extract_page_text(999999)
