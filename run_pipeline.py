"""CLI runner for the Vehicle Specification Extraction pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Optional
from tabulate import tabulate

from src.config import default_config
from src.parser.pdf_loader import PDFLoader
from src.chunking.chunker import SemanticChunker
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_store import VectorStore
from src.extraction.extractor import SpecificationExtractor
from src.extraction.llm_client import get_llm_client
from src.utils.exporter import export_to_json, export_to_csv
from src.utils.logger import setup_logger

logger = setup_logger("pipeline_runner")

def get_or_build_index(
    pdf_path: Path,
    cache_path: Path,
    reindex: bool = False,
    max_pages: Optional[int] = None
) -> VectorStore:
    """Loads existing index from cache or builds a new one from the PDF manual."""
    if cache_path.exists() and not reindex:
        logger.info(f"Loading cached vector index from {cache_path}")
        return VectorStore.load(cache_path)

    page_desc = f"all {max_pages or 852} pages"
    logger.info(f"Parsing PDF manual from {pdf_path} (indexing {page_desc})...")
    loader = PDFLoader(pdf_path)
    pages = loader.load_pages(start_page=1, end_page=max_pages)
    logger.info(f"Extracted {len(pages)} pages successfully.")

    logger.info("Chunking document sections and tables...")
    chunker = SemanticChunker(
        chunk_size=default_config.chunk_size,
        chunk_overlap=default_config.chunk_overlap
    )
    chunks = chunker.chunk_document(pages)
    logger.info(f"Generated {len(chunks)} structured text chunks.")

    logger.info("Building embeddings and vector store index...")
    embedder = TextEmbedder()
    vector_store = VectorStore(embedder=embedder)
    vector_store.build_index(chunks)

    vector_store.save(cache_path)
    logger.info(f"Vector index cached successfully to {cache_path}")
    return vector_store


def main():
    parser = argparse.ArgumentParser(description="Vehicle Specification Extraction Pipeline (RAG + LLM)")
    parser.add_argument("--query", "-q", type=str, default="Torque for brake caliper bolts",
                        help="Specification query (e.g. 'Torque for brake caliper bolts')")
    parser.add_argument("--pages", "-p", type=int, default=None,
                        help="Number of pages to index from manual (default: all pages)")
    parser.add_argument("--reindex", action="store_true",
                        help="Force rebuild the vector index cache")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                        help="Output file format (json or csv)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to save output results")
    args = parser.parse_args()

    cache_file = default_config.cache_dir / "service_manual_index.pkl"
    vector_store = get_or_build_index(
        pdf_path=default_config.pdf_path,
        cache_path=cache_file,
        reindex=args.reindex,
        max_pages=args.pages
    )

    extractor = SpecificationExtractor(vector_store=vector_store)
    logger.info(f"Executing extraction for query: '{args.query}'")
    result = extractor.extract(args.query)

    print("\n" + "=" * 70)
    print(f"QUERY: {result.query}")
    print(f"STATUS: {result.status} | TIME: {result.execution_time_seconds}s | CHUNKS RETRIEVED: {result.retrieved_chunks_count}")
    print(f"PAGES REFERENCED: {result.pages_referenced}")
    print("=" * 70)

    if result.specifications:
        table_rows = []
        for s in result.specifications:
            table_rows.append([
                s.component,
                s.spec_type,
                s.value,
                s.unit,
                f"{s.secondary_value} {s.secondary_unit}" if s.secondary_value else "-",
                f"Page {s.source_page}" if s.source_page else "-"
            ])
        print(tabulate(
            table_rows,
            headers=["Component", "Type", "Value", "Unit", "Secondary Unit", "Source"],
            tablefmt="grid"
        ))
    else:
        print("No specifications extracted.")

    output_path = Path(args.output) if args.output else default_config.output_dir / f"specs_{result.status}.{args.format}"
    if args.format == "csv":
        export_to_csv(result, output_path)
    else:
        export_to_json(result, output_path)

    print(f"\nStructured result exported to: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
