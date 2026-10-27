"""Orchestrator for Retrieval-Augmented Generation (RAG) specification extraction."""

import time
import json
import re
from typing import List, Optional
from ..schemas import SpecificationItem, ExtractionResult
from ..indexing.vector_store import VectorStore
from .llm_client import BaseLLMClient, get_llm_client
from .prompts import format_extraction_prompt


class SpecificationExtractor:
    """Orchestrates document search, prompt formatting, LLM querying, and schema validation."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: Optional[BaseLLMClient] = None,
        top_k: int = 5,
        threshold: float = 0.05
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or get_llm_client()
        self.top_k = top_k
        self.threshold = threshold

    def extract(self, query: str, top_k: Optional[int] = None) -> ExtractionResult:
        """
        Executes an end-to-end extraction query against the service manual vector store.
        """
        start_time = time.time()
        k = top_k or self.top_k

        # 1. Retrieve most relevant context chunks
        search_hits = self.vector_store.search(query, top_k=k, threshold=self.threshold)

        if not search_hits:
            return ExtractionResult(
                query=query,
                specifications=[],
                retrieved_chunks_count=0,
                pages_referenced=[],
                execution_time_seconds=round(time.time() - start_time, 3),
                status="no_matches_found",
                message="No relevant manual excerpts found matching the query."
            )

        # 2. Format context and collect page provenance
        context_parts = []
        pages_seen = []
        for chunk, score in search_hits:
            header = f"[Page {chunk.page_number}"
            if chunk.section_title:
                header += f" | {chunk.section_title}"
            header += f" | Relevance: {score:.2f}]"
            context_parts.append(f"{header}\n{chunk.text}")
            if chunk.page_number not in pages_seen:
                pages_seen.append(chunk.page_number)

        combined_context = "\n\n---\n\n".join(context_parts)
        prompt = format_extraction_prompt(query=query, retrieved_context=combined_context)

        # 3. Query LLM
        raw_response = self.llm_client.generate(prompt)

        # 4. Clean & Parse JSON response
        spec_items: List[SpecificationItem] = []
        try:
            # Strip markdown code fences if present (e.g. ```json ... ```)
            json_text = raw_response.strip()
            if json_text.startswith("```"):
                json_text = re.sub(r"^```[a-zA-Z]*\n", "", json_text)
                json_text = re.sub(r"\n```$", "", json_text)
            parsed_data = json.loads(json_text)

            if isinstance(parsed_data, list):
                for item in parsed_data:
                    try:
                        # Auto-assign source page if available from first retrieved chunk
                        if "source_page" not in item or item["source_page"] is None:
                            item["source_page"] = pages_seen[0] if pages_seen else None
                        spec_items.append(SpecificationItem(**item))
                    except Exception as val_err:
                        # Log validation error but proceed with valid items
                        continue

        except Exception as json_err:
            return ExtractionResult(
                query=query,
                specifications=[],
                retrieved_chunks_count=len(search_hits),
                pages_referenced=pages_seen,
                execution_time_seconds=round(time.time() - start_time, 3),
                status="parse_error",
                message=f"Failed to parse LLM response: {str(json_err)}"
            )

        elapsed = round(time.time() - start_time, 3)

        return ExtractionResult(
            query=query,
            specifications=spec_items,
            retrieved_chunks_count=len(search_hits),
            pages_referenced=pages_seen,
            execution_time_seconds=elapsed,
            status="success" if spec_items else "empty_result",
            message=f"Extracted {len(spec_items)} specification(s)" if spec_items else "No explicit specification found for query."
        )
