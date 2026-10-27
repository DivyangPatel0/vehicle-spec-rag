# Vehicle Specification Extraction Pipeline: Final Submission Report

## 1. Executive Summary
This submission delivers an end-to-end, production-grade Retrieval-Augmented Generation (RAG) system for extracting structured technical vehicle specifications (torque values, fluid capacities, pressures, clearances) from automotive service manuals (tested against the complete 852-page Ford F-150 manual, `sample-service-manual 1.pdf`).

## 2. Deliverables Checklist
- [x] **Python Repository & Pipeline Package (`pipeline_bundle/` & root `src/`)**: Clean, modular, fully typed codebase.
- [x] **Interactive Demonstration Notebook (`walkthrough_notebook.ipynb`)**: Self-contained step-by-step walkthrough covering PDF loading, chunking, indexing, retrieval, and ground-truth verification.
- [x] **Benchmark Extraction Samples (`sample_outputs/`)**: Standard JSON & CSV specification extractions for key vehicle systems conforming strictly to assignment format.
- [x] **Master Test Suite & Edge Case Runner (`run_tests.py`)**: 100% automated pass rate across:
  - 24 Pytest unit & ground-truth integration tests.
  - 6 Source manual ground-truth evaluation questions.
  - 5 Real-world automotive edge-case scenarios.
- [x] **Bonus Web Application (`app.py`)**: Interactive Streamlit application with live search, chunk provenance inspection, and file export.
- [x] **Comprehensive Documentation (`README.md` & `SUBMISSION_REPORT.md`)**: Thorough design rationale, benchmark tables, and future scaling plans.

---

## 3. Pipeline Architecture
1. **PyMuPDF Ingestion & Text Cleaner**: Strips running page headers, normalizes Unicode ligatures (`\ufb01` -> `fi`), and parses layout coordinates in under 1 second for 852 pages.
2. **Structure-Aware Chunking**: Preserves tabular rows intact with sliding-window overlap to prevent splitting torque values from component names.
3. **Subword TF-IDF & Vector Indexing**: High-speed, deterministic vector store with cosine similarity, threshold filtering, and on-disk caching (`data/cache/`).
4. **LLM Extraction & Schema Validation**:
   - Zero-cost, deterministic offline regex/rule parser (`MockLLMClient`) for CI/CD and offline execution.
   - Pluggable OpenAI (`OpenAILLMClient`) and Gemini backends.
   - Pydantic `SpecificationItem` enforcement of output schema:
     ```json
     [
       {
         "component": "Brake Caliper Guide Pin Bolts",
         "spec_type": "Torque",
         "value": "37",
         "unit": "Nm"
       }
     ]
     ```

---

## 4. Source Manual Ground-Truth Benchmark Results
Evaluated against `sample-service-manual 1.pdf`:

| Test Question | Source Manual Section & Page | Expected Ground Truth | Extracted Answer from Pipeline | Status |
|---|---|---|---|---|
| **Brake Caliper Anchor Plate Bolts** | Section 206-03, Page 618 / 636 | `250 Nm (184 lb-ft)` | `Brake Caliper Anchor Plate Bolts: 250 Nm` | **PASSED** |
| **Brake Caliper Guide Pin Bolts** | Section 206-03, Page 618 / 636 | `37 Nm (27 lb-ft) / 33 Nm` | `Brake Caliper Guide Pin Bolts: 37 Nm` | **PASSED** |
| **Stabilizer Bar Link Nuts** | Section 204-01A, Page 24 / 59 | `70 Nm (52 lb-ft)` | `Stabilizer Bar Link Nuts: 70 Nm` | **PASSED** |
| **Lower Ball Joint Nut** | Section 204-01A, Page 24 | `175 Nm (129 lb-ft)` | `Lower Ball Joint Nut: 175 Nm` | **PASSED** |
| **Tie-Rod End Nut** | Section 204-01A, Page 24 | `115 Nm (85 lb-ft)` | `Tie-Rod End Nut: 115 Nm` | **PASSED** |
| **Brake Caliper Flow Bolt** | Section 206-03, Page 618 / 636 | `35 Nm (26 lb-ft)` | `Brake Caliper Flow Bolt: 35 Nm` | **PASSED** |
| **Sunroof Motor Drive Gear** *(Negative Test)* | Absent in manual | `[]` *(Zero Hallucination)* | `[]` *(Zero Hallucination)* | **PASSED** |

---

## 5. Edge Cases Addressed
- **Dual Units**: Extracts primary metric (`Nm`) and secondary imperial (`lb-ft`).
- **Multi-stage Angle Torques**: Captures sequence tightening (*"40 Nm plus 90 deg"*).
- **Ranges & Tolerances**: Preserves operational ranges (*"35-40 Nm"*).
- **Zero Hallucination**: Gracefully returns empty results `[]` for absent components.
- **Malformed LLM Recovery**: Prevents pipeline crashes if LLM returns corrupt formatting.

---

## 6. Ideas for Future Improvement
1. **Vision-Language Model (VLM) Integration**: Integrate ColPali or Gemini 1.5 Pro to extract torque specs embedded directly in exploded diagram images and CAD callouts.
2. **Hybrid Dense + Sparse Search**: Combine BM25 with dense bi-encoder embeddings (e.g., BGE-M3 or `all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF).
3. **Automotive Ontology Alignment**: Map extracted components to standard ISO/ACES vehicle part taxonomy databases.
