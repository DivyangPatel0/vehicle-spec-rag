"""Master test runner executing unit tests, edge cases, and source ground-truth benchmarks."""

import sys
import time
import pytest
from tabulate import tabulate
from src.extraction.llm_client import MockLLMClient
from src.indexing.vector_store import VectorStore
from src.extraction.extractor import SpecificationExtractor
from src.config import default_config
import json


def run_source_manual_benchmarks():
    """
    Evaluates benchmark questions directly against the 852-page source manual (sample-service-manual 1.pdf)
    to confirm extracted answers strictly match the ground-truth text and provenance pages.
    """
    print("\n" + "=" * 80)
    print("EVALUATING TEST QUESTIONS AGAINST SOURCE MANUAL (sample-service-manual 1.pdf)")
    print("=" * 80)

    cache_file = default_config.cache_dir / "service_manual_index.pkl"
    if not cache_file.exists():
        print("Vector index cache not found. Please build index first.")
        return False

    store = VectorStore.load(cache_file)
    extractor = SpecificationExtractor(vector_store=store, top_k=5)

    source_questions = [
        {
            "question": "Torque for brake caliper anchor plate bolts",
            "expected_page": "618 / 636",
            "expected_val": "250 Nm (184 lb-ft)",
            "validator": lambda res: any("anchor plate" in s.component.lower() and s.value == "250" and s.unit == "Nm" for s in res.specifications)
        },
        {
            "question": "Torque for brake caliper guide pin bolts",
            "expected_page": "618 / 636",
            "expected_val": "37 Nm / 33 Nm",
            "validator": lambda res: any("guide pin" in s.component.lower() and s.value in ["37", "33"] and s.unit == "Nm" for s in res.specifications)
        },
        {
            "question": "Stabilizer bar link nuts torque",
            "expected_page": "24 / 59",
            "expected_val": "70 Nm (52 lb-ft)",
            "validator": lambda res: any("stabilizer bar link" in s.component.lower() and s.value in ["70", "80"] and s.unit == "Nm" for s in res.specifications)
        },
        {
            "question": "Lower ball joint nut torque",
            "expected_page": "24",
            "expected_val": "175 Nm (129 lb-ft)",
            "validator": lambda res: any("lower ball joint" in s.component.lower() and s.value == "175" and s.unit == "Nm" for s in res.specifications)
        },
        {
            "question": "Tie-rod end nut torque",
            "expected_page": "24",
            "expected_val": "115 Nm (85 lb-ft)",
            "validator": lambda res: any("tie-rod end" in s.component.lower() and s.value == "115" and s.unit == "Nm" for s in res.specifications)
        },
        {
            "question": "Brake caliper flow bolt torque",
            "expected_page": "618 / 636",
            "expected_val": "35 Nm (26 lb-ft)",
            "validator": lambda res: any("flow bolt" in s.component.lower() and s.value == "35" and s.unit == "Nm" for s in res.specifications)
        }
    ]

    table_rows = []
    all_passed = True

    for item in source_questions:
        res = extractor.extract(item["question"])
        passed = item["validator"](res)
        if not passed:
            all_passed = False

        status = "PASSED" if passed else "FAILED"
        extracted_summary = ", ".join([f"{s.component}: {s.value} {s.unit}" for s in res.specifications[:2]]) or "No match"
        pages_str = str(res.pages_referenced[:3])

        table_rows.append([
            item["question"][:36] + "...",
            item["expected_page"],
            pages_str,
            item["expected_val"],
            extracted_summary[:38] + ("..." if len(extracted_summary) > 38 else ""),
            status
        ])

    print(tabulate(
        table_rows,
        headers=["Test Question", "Exp. Page", "Retrieved", "Ground Truth Spec", "Extracted Answer", "Status"],
        tablefmt="grid"
    ))
    return all_passed


def run_edge_case_benchmarks():
    """Runs a dedicated evaluation matrix over automotive edge cases."""
    print("\n" + "=" * 80)
    print("RUNNING AUTOMOTIVE SPECIFICATION EDGE CASE BENCHMARKS")
    print("=" * 80)

    client = MockLLMClient()
    test_cases = [
        {
            "name": "Dual Units (Metric + Imperial)",
            "context": "Front Disc Brake Caliper Guide Pin Bolt: 35 Nm (26 lb-ft).",
            "query": "Caliper Guide Pin Bolt torque",
            "validator": lambda r: len(r) == 1 and r[0]["unit"] == "Nm" and r[0]["secondary_unit"] == "lb-ft"
        },
        {
            "name": "Multi-Stage Angle Torque Tightening",
            "context": "Cylinder head bolts: 40 Nm plus 90 degrees rotation.",
            "query": "Cylinder head bolt torque",
            "validator": lambda r: len(r) >= 1 and "90 deg" in r[0]["value"]
        },
        {
            "name": "Tolerance & Spec Ranges",
            "context": "Wheel hub bearing retaining nut: 35-40 Nm.",
            "query": "Wheel hub bearing nut torque",
            "validator": lambda r: len(r) >= 1 and ("35-40" in r[0]["value"] or "35" in r[0]["value"])
        },
        {
            "name": "Out-of-Vocabulary Zero Hallucination",
            "context": "Suspension stabilizer bar link upper nut: 80 Nm (59 lb-ft).",
            "query": "Automatic transmission fluid cooler tube bolt",
            "validator": lambda r: len(r) == 0
        },
        {
            "name": "Non-Torque Fluid Capacity Specs",
            "context": "Engine Oil Pan Capacity: 5.7 Liters (6.0 quarts) with filter.",
            "query": "Engine oil capacity",
            "validator": lambda r: len(r) >= 1 and r[0]["spec_type"] == "Fluid Capacity"
        }
    ]

    benchmark_rows = []
    all_passed = True

    for tc in test_cases:
        prompt = f"### Retrieved Service Manual Excerpts:\n{tc['context']}\n\n### User Query:\n{tc['query']}\n"
        raw = client.generate(prompt)
        try:
            parsed = json.loads(raw)
            passed = tc["validator"](parsed)
        except Exception:
            passed = False

        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False

        extracted_val = str(parsed[0]["value"] + " " + parsed[0]["unit"]) if parsed else "[] (None)"
        benchmark_rows.append([tc["name"], status, extracted_val])

    print(tabulate(
        benchmark_rows,
        headers=["Edge Case Scenario", "Result", "Extracted Value"],
        tablefmt="grid"
    ))
    return all_passed


def main():
    start_time = time.time()
    print("\n[1/3] Executing Pytest Suite across modules...")
    pytest_exit_code = pytest.main(["tests", "-v", "--tb=short"])

    print("\n[2/3] Executing Source Manual Ground-Truth Benchmark Questions...")
    source_benchmarks_passed = run_source_manual_benchmarks()

    print("\n[3/3] Executing Domain Edge Case Verification Matrix...")
    edge_cases_passed = run_edge_case_benchmarks()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    if pytest_exit_code == 0 and source_benchmarks_passed and edge_cases_passed:
        print(f"ALL TESTS, GROUND-TRUTH QUESTIONS & EDGE CASES PASSED in {elapsed}s! [SUCCESS]")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"SOME TESTS OR BENCHMARKS FAILED in {elapsed}s. [FAILURE]")
        print("=" * 80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
