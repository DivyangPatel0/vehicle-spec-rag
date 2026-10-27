"""Streamlit Web UI for Interactive Vehicle Specification Extraction (Bonus UI Feature)."""

import streamlit as st
import json
import pandas as pd
from pathlib import Path

from src.config import default_config
from src.parser.pdf_loader import PDFLoader
from src.chunking.chunker import SemanticChunker
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_store import VectorStore
from src.extraction.extractor import SpecificationExtractor
from src.extraction.llm_client import get_llm_client
from src.utils.exporter import export_to_json, export_to_csv

st.set_page_config(
    page_title="Vehicle Spec Extractor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Automotive Vehicle Specification Extraction")
st.markdown(
    "Extract mechanical specifications (torques, fluid capacities, pressures) from the service manual using RAG."
)

@st.cache_resource(show_spinner="Loading and caching manual index...")
def load_manual_index():
    cache_file = default_config.cache_dir / "service_manual_index.pkl"
    if cache_file.exists():
        return VectorStore.load(cache_file)
    
    loader = PDFLoader(default_config.pdf_path)
    pages = loader.load_pages()
    chunker = SemanticChunker(
        chunk_size=default_config.chunk_size,
        chunk_overlap=default_config.chunk_overlap
    )
    chunks = chunker.chunk_document(pages)
    embedder = TextEmbedder()
    store = VectorStore(embedder=embedder)
    store.build_index(chunks)
    store.save(cache_file)
    return store

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
preset_queries = [
    "Torque for brake caliper bolts",
    "Front stabilizer bar link upper nut torque",
    "Wheel hub bearing retaining nut torque",
    "Engine oil pan capacity and drain plug torque",
    "Custom Query..."
]
selected_preset = st.sidebar.selectbox("Preset Example Queries", preset_queries)

if selected_preset == "Custom Query...":
    query_input = st.text_input("Enter vehicle specification query:", "Torque for brake caliper bolts")
else:
    query_input = st.text_input("Enter vehicle specification query:", selected_preset)

col_k, col_thresh = st.sidebar.columns(2)
with col_k:
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=5)
with col_thresh:
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

run_button = st.button("🔍 Extract Specifications", type="primary")

if run_button or query_input:
    try:
        store = load_manual_index()
        extractor = SpecificationExtractor(vector_store=store, top_k=top_k, threshold=threshold)
        
        with st.spinner("Retrieving manual excerpts and extracting specifications..."):
            result = extractor.extract(query_input)

        st.subheader("📊 Extraction Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", result.status.upper())
        m2.metric("Extracted Items", len(result.specifications))
        m3.metric("Chunks Retrieved", result.retrieved_chunks_count)
        m4.metric("Latency", f"{result.execution_time_seconds}s")

        if result.specifications:
            clean_list = result.to_clean_list()
            try:
                st.dataframe(df, width="stretch")
            except TypeError:
                st.dataframe(df, use_container_width=True)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(clean_list, indent=2),
                    file_name="extracted_specs.json",
                    mime="application/json"
                )
            with col_d2:
                st.download_button(
                    label="📥 Download CSV",
                    data=df.to_csv(index=False),
                    file_name="extracted_specs.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No explicit specifications matched this query in the retrieved sections.")

        with st.expander("📄 View Retrieved Context Excerpts & Provenance"):
            search_hits = store.search(query_input, top_k=top_k, threshold=threshold)
            for idx, (chunk, score) in enumerate(search_hits, start=1):
                st.markdown(f"**Chunk #{idx}** — Page: `{chunk.page_number}` | Score: `{score:.3f}` | Section: `{chunk.section_title or 'General'}`")
                st.code(chunk.text, language="text")

    except Exception as e:
        st.error(f"Error during extraction: {e}")
