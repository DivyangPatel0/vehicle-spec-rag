"""PyMuPDF loader for automotive PDF service manuals."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import pymupdf

from .text_cleaner import clean_extracted_text


class PDFLoader:
    """Extracts text and structural elements from PDF documents."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")
        self._doc = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Opens the PDF document."""
        if self._doc is None or self._doc.is_closed:
            self._doc = pymupdf.open(str(self.pdf_path))
        return self

    def close(self):
        """Closes the PDF document handle."""
        if self._doc is not None and not self._doc.is_closed:
            self._doc.close()

    @property
    def total_pages(self) -> int:
        """Returns total number of pages in the PDF."""
        self.open()
        return len(self._doc)

    def extract_page_text(self, page_number: int, clean: bool = True) -> Dict[str, Any]:
        """
        Extracts text from a single 1-indexed page.
        
        Args:
            page_number: 1-based page index.
            clean: Whether to apply ligature and noise normalization.
        """
        self.open()
        if page_number < 1 or page_number > len(self._doc):
            raise IndexError(f"Page number {page_number} is out of bounds (1-{len(self._doc)})")

        page = self._doc[page_number - 1]
        raw_text = page.get_text("text")

        # Detect potential section title from top blocks
        blocks = page.get_text("blocks")
        detected_title = None
        if blocks:
            for b in blocks[:2]:
                text_block = b[4].strip()
                if text_block and len(text_block) < 80:
                    detected_title = text_block
                    break

        cleaned_text = clean_extracted_text(raw_text) if clean else raw_text

        return {
            "page_number": page_number,
            "raw_text": raw_text,
            "text": cleaned_text,
            "section_title": detected_title,
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }

    def load_pages(
        self,
        start_page: int = 1,
        end_page: Optional[int] = None,
        clean: bool = True
    ) -> List[Dict[str, Any]]:
        """Loads a slice of pages into memory."""
        self.open()
        end = end_page if end_page is not None else len(self._doc)
        pages = []
        for p in range(start_page, end + 1):
            pages.append(self.extract_page_text(p, clean=clean))
        return pages

    def iter_pages(
        self,
        start_page: int = 1,
        end_page: Optional[int] = None,
        clean: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """Generator yielding pages one-by-one to conserve memory on large PDFs."""
        self.open()
        end = end_page if end_page is not None else len(self._doc)
        for p in range(start_page, end + 1):
            yield self.extract_page_text(p, clean=clean)
