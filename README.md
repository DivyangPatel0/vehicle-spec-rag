# Automotive Vehicle Specification Extraction System (RAG + LLM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-modular-orange.svg)](#repository-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** pipeline designed to extract structured mechanical specifications (torque values, fluid capacities, pressures, clearances) from automotive service manuals (e.g., Ford F-150 852-page manual) using LLMs and vector retrieval.

---

## 📋 Table of Contents
- [Objective & Scope](#objective--scope)
- [Repository Architecture](#repository-architecture)
- [Tools & Technologies Used](#tools--technologies-used)
- [Pipeline Architecture & Design](#pipeline-architecture--design)
- [Installation & Quick Start](#installation--quick-start)
  - [1. CLI Pipeline Runner](#1-cli-pipeline-runner)
  - [2. Interactive Streamlit Web UI (Bonus)](#2-interactive-streamlit-web-ui-bonus)
  - [3. Master Test Runner & Edge Case Suite](#3-master-test-runner--edge-case-suite)
  - [4. Walkthrough Jupyter Notebook](#4-walkthrough-jupyter-notebook)
- [Edge Cases Handled](#edge-cases-handled)
- [Output Format & Deliverables](#output-format--deliverables)
- [Ideas for Future Improvement](#ideas-for-future-improvement)

---

## 🎯 Objective & Scope
Automotive service manuals are dense, multi-hundred-page technical PDF documents containing critical maintenance and repair data. This project implements a robust system that:
1. **Parses & Cleans PDF Text**: Extracts text, normalizes ligatures, strips recurring headers, and retains structural alignment.
2. **Chunks with Context & Table Awareness**: Preserves specification rows so torque values and component names stay coupled.
3. **Embeds & Indexes**: Performs semantic search using cosine similarity over normalized vector embeddings with disk caching.
4. **Extracts via LLM + Pydantic**: Queries an LLM with strict few-shot prompt constraints and validates schema output into standard JSON/CSV formats.
5. **Handles Automotive Edge Cases**: Evaluates multi-stage angle torques, dual units (metric + imperial), tolerance ranges, and out-of-vocabulary queries without hallucinations.

---

## 📂 Repository Architecture

```
predii/
├── Assignment-specs-extraction 1.pdf   # Original assignment specification
├── sample-service-manual 1.pdf         # 852-page automotive service manual
├── README.md                           # Comprehensive documentation & design guide
├── requirements.txt                    # Pinned project dependencies
├── pyproject.toml                      # Tooling configuration (pytest, ruff)
├── run_tests.py                        # Master test runner with edge case verification
├── run_pipeline.py                     # CLI pipeline runner (query -> retrieve -> extract -> export)
├── app.py                              # Streamlit Web UI (Bonus UI feature)
│
├── src/                                # Core modular Python package
│   ├── config.py                       # Centralized configuration & settings
│   ├── schemas.py                      # Pydantic models for specifications & metadata
│   ├── parser/                         # PDF text & structure extraction (PyMuPDF)
│   │   ├── pdf_loader.py               # Memory-efficient PDF page loader
│   │   └── text_cleaner.py             # Ligature translation & header cleaner
│   ├── chunking/                       # Semantic and table-aware chunking
│   │   └── chunker.py                  # Structure-aware chunker
│   ├── indexing/                       # Embeddings & persistent vector indexing
│   │   ├── embedder.py                 # Subword TF-IDF / transformer embedder
│   │   └── vector_store.py             # Cosine similarity store with disk serialization
│   ├── extraction/                     # RAG prompt engine, LLM clients, schema validation
│   │   ├── prompts.py                  # Domain system prompt & few-shot examples
│   │   ├── llm_client.py               # Pluggable backends (Mock, OpenAI, Gemini)
│   │   └── extractor.py                # Pipeline orchestrator
│   └── utils/                          # Exporters & deliverables packager
│       ├── exporter.py                 # Strict assignment JSON & CSV serializers
│       ├── logger.py                   # Structured console logging
│       └── package_deliverables.py     # Automated submission packaging
│
├── tests/                              # Comprehensive test suite
│   ├── test_pdf_loader.py              # Parsing, error handling & ligature normalization
│   ├── test_chunking.py                # Table preservation & boundary overlap
│   ├── test_retrieval.py               # Top-K relevance & thresholding
│   ├── test_extraction_edge_cases.py   # Dual units, stage torques, missing values
│   └── test_schemas.py                 # Schema validation & type coercion
│
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb   # End-to-end reproducible walkthrough notebook
│
└── deliverables/                       # Automated final deliverable package
    ├── pipeline_bundle/                # Core pipeline code & configs
    ├── sample_outputs/                 # Extracted JSON and CSV files for benchmark queries
    │   ├── brake_caliper_specs.json
    │   ├── suspension_stabilizer_specs.json
    │   └── wheel_bearing_specs.json
    ├── walkthrough_notebook.ipynb      # Clean standalone demonstration notebook
    └── SUBMISSION_REPORT.md            # Comprehensive documentation & evaluation report
```

---

## 🛠️ Tools & Technologies Used

| Stage | Tool / Library | Rationale |
|---|---|---|
| **PDF Parsing** | `PyMuPDF` (`fitz`) | Up to 10x faster than `pdfminer` / `pypdf`, preserves layout coordinates, and reliably handles dense 800+ page manuals. |
| **Data Validation** | `Pydantic v2` | Strong runtime typing, schema guarantees, and automatic sanitization of extracted specification fields. |
| **Chunking** | Custom Structure-Aware Chunker | Naive character splitters break tables mid-row. Our chunker detects numerical units (`Nm`, `lb-ft`) to keep rows intact. |
| **Embedding & Vector Search** | `scikit-learn` & `numpy` | Subword n-gram vectorization with L2 normalization provides sub-millisecond retrieval on technical vocabulary with zero cloud dependencies. Supports drop-in sentence-transformers. |
| **LLM Orchestration** | Pluggable Engine | Includes `MockLLMClient` for 100% offline, deterministic CI/evaluation, alongside `OpenAILLMClient` and `GeminiLLMClient`. |
| **User Interface** | `Streamlit` | Interactive dashboard enabling manual querying, confidence inspection, and instant download of JSON/CSV. |
| **Testing** | `pytest` + Custom Test Runner | Automated suite testing unit functions, parsing edge cases, and end-to-end extraction accuracy. |

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
Ensure Python 3.10+ is installed:
```bash
python --version
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 1. CLI Pipeline Runner (`run_pipeline.py`)
Run an end-to-end extraction query directly from the terminal:
```bash
# Query torque for brake caliper bolts
python run_pipeline.py --query "Torque for brake caliper bolts"

# Query suspension components
python run_pipeline.py --query "Front stabilizer bar link upper nut torque" --format csv
```

Example CLI Output:
```text
======================================================================
QUERY: Torque for brake caliper bolts
STATUS: success | TIME: 0.045s | CHUNKS RETRIEVED: 5
PAGES REFERENCED: [12, 14, 18]
======================================================================
+------------------------------------+--------+---------+--------+------------------+----------+
| Component                          | Type   | Value   | Unit   | Secondary Unit   | Source   |
+====================================+========+=========+========+==================+==========+
| Front Disc Brake Caliper Guide Pin | Torque | 35      | Nm     | 26 lb-ft         | Page 12  |
| Caliper Anchor Plate Bolts         | Torque | 175     | Nm     | 129 lb-ft        | Page 14  |
+------------------------------------+--------+---------+--------+------------------+----------+

Structured result exported to: outputs/specs_success.json
```

---

### 2. Interactive Streamlit Web UI (Bonus)
Launch the interactive web application:
```bash
streamlit run app.py
```
**Features**:
- Interactive query input with preset automotive queries.
- Real-time specification extraction table.
- View retrieved chunk provenance and relevance scores.
- One-click export to formatted JSON and CSV.

---

### 3. Master Test Runner & Edge Case Suite
Run the master test runner to execute all unit tests and verify the domain edge cases:
```bash
python run_tests.py
```

Expected output:
```text
[1/2] Executing Pytest Suite across modules...
============================= 18 passed in 0.20s ==============================

[2/2] Executing Domain Edge Case Benchmark Verification...
======================================================================
RUNNING AUTOMOTIVE SPECIFICATION EDGE CASE BENCHMARKS
======================================================================
+--------------------------------------+----------+-------------------+
| Edge Case Scenario                   | Result   | Extracted Value   |
+======================================+==========+===================+
| Dual Units (Metric + Imperial)       | PASSED   | 35 Nm             |
| Multi-Stage Angle Torque Tightening  | PASSED   | 40 + 90 deg Nm    |
| Tolerance & Spec Ranges              | PASSED   | 35-40 Nm          |
| Out-of-Vocabulary Zero Hallucination | PASSED   | [] (None)         |
| Non-Torque Fluid Capacity Specs      | PASSED   | 5.7 Liters        |
+--------------------------------------+----------+-------------------+
======================================================================
ALL TESTS & EDGE CASE BENCHMARKS PASSED in 0.47s! [SUCCESS]
======================================================================
```

---

### 4. Walkthrough Jupyter Notebook
Launch Jupyter to explore the step-by-step pipeline demonstration:
```bash
jupyter notebook notebooks/01_pipeline_walkthrough.ipynb
```

---

## 🔍 Edge Cases Handled

| Scenario | Service Manual Sample | Pipeline Handling |
|---|---|---|
| **Dual Units** | `35 Nm (26 lb-ft)` | Extracts primary metric unit (`value`: "35", `unit`: "Nm") and secondary imperial unit (`secondary_value`: "26", `secondary_unit`: "lb-ft"). |
| **Multi-Stage Angle Torque** | `40 Nm plus 90 degrees rotation` | Accurately extracts `40 + 90 deg` preserving the two-phase tightening procedure. |
| **Tolerance Ranges** | `35-40 Nm` or `0.02 - 0.05 mm` | Captures operational ranges without rounding or discarding limits. |
| **PDF Typographical Ligatures** | `Speci\ufb01cation \ufb02uid` | Unicode NFKC and ligature mapping replaces `\ufb01` with `fi` and `\ufb02` with `fl`. |
| **Out-of-Vocabulary Queries** | Part not in manual | Clean fallback returning `[]` with explanatory status; avoids LLM hallucinations. |
| **Corrupt LLM Output** | Malformed JSON string | Schema validator catches parse errors gracefully, logging details without terminating the application. |

---

## 📤 Output Format & Deliverables

Standardized JSON output strictly adheres to the assignment specification:
```json
[
  {
    "component": "Brake Caliper Guide Pin Bolt",
    "spec_type": "Torque",
    "value": "35",
    "unit": "Nm"
  },
  {
    "component": "Caliper Anchor Plate Bolts",
    "spec_type": "Torque",
    "value": "175",
    "unit": "Nm"
  }
]
```

To package all final submission assets into `deliverables/`:
```bash
python -m src.utils.package_deliverables
```

---

## 💡 Ideas for Future Improvement
1. **Vision-Language Models (VLM) for Diagrams**: Many automotive torque values appear inside exploded assembly diagrams (callout balloons with arrows). Integrating ColPali or Gemini 1.5 Pro with bounding box detection would unlock diagram-embedded specifications.
2. **Hybrid Search (Dense + Sparse RRF)**: Combine dense semantic embeddings with BM25 keyword matching via Reciprocal Rank Fusion (RRF) to maximize recall for obscure part numbers.
3. **Automotive Part Taxonomy / Ontology Linking**: Match extracted component strings against standard automotive industry ontologies (e.g., ACES / PIES standard part terminology).
4. **Table Structure Recognition**: Integrate Table Transformer or LayoutLMv3 for multi-column specification tables that span across page breaks.
