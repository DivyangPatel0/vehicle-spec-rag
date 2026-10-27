"""Structure-aware chunker for automotive service manuals."""

import re
from typing import List, Dict, Any, Optional
from ..schemas import DocumentChunk

# Common units in service manuals indicating specification tables or rows
SPEC_UNITS_REGEX = re.compile(
    r"\b(Nm|N-m|N·m|lb-ft|lb-in|ft-lb|in-lb|psi|kPa|bar|mm|in|quarts|liters?|L|deg|degrees)\b",
    re.IGNORECASE
)

class SemanticChunker:
    """
    Chunks document text into structured chunks while preserving
    specification tables, bullet points, and section titles.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        preserve_tables: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables

    def chunk_page(self, page_data: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunks text from a single page dictionary (from PDFLoader).
        """
        page_num = page_data["page_number"]
        text = page_data.get("text", "").strip()
        section_title = page_data.get("section_title")

        if not text:
            return []

        # Split text into structural units (paragraphs or table blocks)
        raw_blocks = text.split("\n\n")
        blocks = []
        for b in raw_blocks:
            b_stripped = b.strip()
            if b_stripped:
                # If a block is large, break into individual lines while keeping spec lines intact
                if len(b_stripped) > self.chunk_size:
                    lines = b_stripped.splitlines()
                    current_line_acc = []
                    current_len = 0
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        current_line_acc.append(line)
                        current_len += len(line) + 1
                        if current_len >= self.chunk_size // 2:
                            blocks.append("\n".join(current_line_acc))
                            current_line_acc = []
                            current_len = 0
                    if current_line_acc:
                        blocks.append("\n".join(current_line_acc))
                else:
                    blocks.append(b_stripped)

        # Build chunks by aggregating blocks with sliding window
        chunks: List[DocumentChunk] = []
        current_chunk_blocks: List[str] = []
        current_length = 0
        chunk_idx = 1

        for block in blocks:
            block_len = len(block)
            is_spec = bool(SPEC_UNITS_REGEX.search(block))

            if current_length + block_len > self.chunk_size and current_chunk_blocks:
                # Emit current chunk
                chunk_text = "\n\n".join(current_chunk_blocks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"p{page_num}_c{chunk_idx}",
                        page_number=page_num,
                        text=chunk_text,
                        section_title=section_title,
                        is_table_row=bool(SPEC_UNITS_REGEX.search(chunk_text))
                    )
                )
                chunk_idx += 1

                # Overlap: keep the trailing block(s) up to chunk_overlap characters
                overlap_blocks = []
                overlap_len = 0
                for prev_block in reversed(current_chunk_blocks):
                    if overlap_len + len(prev_block) <= self.chunk_overlap:
                        overlap_blocks.insert(0, prev_block)
                        overlap_len += len(prev_block)
                    else:
                        break
                current_chunk_blocks = overlap_blocks
                current_length = overlap_len

            current_chunk_blocks.append(block)
            current_length += block_len

        # Flush any remaining text in buffer
        if current_chunk_blocks:
            chunk_text = "\n\n".join(current_chunk_blocks)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"p{page_num}_c{chunk_idx}",
                    page_number=page_num,
                    text=chunk_text,
                    section_title=section_title,
                    is_table_row=bool(SPEC_UNITS_REGEX.search(chunk_text))
                )
            )

        return chunks

    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Chunks a sequence of pages."""
        all_chunks: List[DocumentChunk] = []
        for p in pages:
            all_chunks.extend(self.chunk_page(p))
        return all_chunks
