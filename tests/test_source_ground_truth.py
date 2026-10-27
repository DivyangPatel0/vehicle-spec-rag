"""Ground-truth validation tests evaluating questions directly against sample-service-manual 1.pdf."""

import pytest
from pathlib import Path
from src.config import default_config
from src.indexing.vector_store import VectorStore
from src.extraction.extractor import SpecificationExtractor

MANUAL_PATH = default_config.pdf_path
CACHE_PATH = default_config.cache_dir / "service_manual_index.pkl"


@pytest.fixture(scope="module")
def extractor():
    """Initializes extractor with the cached 852-page service manual index."""
    if not CACHE_PATH.exists():
        pytest.skip("Manual index not cached. Run run_pipeline.py first.")
    store = VectorStore.load(CACHE_PATH)
    return SpecificationExtractor(vector_store=store, top_k=5)


def test_gt_brake_caliper_anchor_plate(extractor):
    """Q1: Verify brake caliper anchor plate bolts torque from manual pages 618/636."""
    res = extractor.extract("Torque for brake caliper anchor plate bolts")
    assert res.status == "success"
    assert any(p in [618, 636, 638, 640] for p in res.pages_referenced)

    # Ground truth: 250 Nm (184 lb-ft)
    matched = [s for s in res.specifications if "anchor plate" in s.component.lower()]
    assert len(matched) > 0
    assert matched[0].value == "250"
    assert matched[0].unit == "Nm"


def test_gt_brake_caliper_guide_pins(extractor):
    """Q2: Verify brake caliper guide pin bolts torque from manual pages 618/636."""
    res = extractor.extract("Torque for brake caliper guide pin bolts")
    assert res.status == "success"
    assert any(p in [618, 636, 652, 628] for p in res.pages_referenced)

    # Ground truth: 37 Nm or 33 Nm
    matched = [s for s in res.specifications if "guide pin" in s.component.lower()]
    assert len(matched) > 0
    assert matched[0].value in ["37", "33"]
    assert matched[0].unit == "Nm"


def test_gt_stabilizer_bar_link_nuts(extractor):
    """Q3: Verify stabilizer bar link nuts torque from manual page 24."""
    res = extractor.extract("Stabilizer bar link nuts torque")
    assert res.status == "success"
    assert any(p in [24, 59, 12, 15] for p in res.pages_referenced)

    # Ground truth: 70 Nm (52 lb-ft) or 80 Nm (SVT Raptor)
    matched = [s for s in res.specifications if "stabilizer bar link" in s.component.lower()]
    assert len(matched) > 0
    assert matched[0].value in ["70", "80"]
    assert matched[0].unit == "Nm"


def test_gt_lower_ball_joint_nut(extractor):
    """Q4: Verify lower ball joint nut torque from manual page 24."""
    res = extractor.extract("Lower ball joint nut torque")
    assert res.status == "success"
    assert any(p in [24, 28, 59] for p in res.pages_referenced)

    # Ground truth: 175 Nm (129 lb-ft)
    matched = [s for s in res.specifications if "lower ball joint" in s.component.lower()]
    assert len(matched) > 0
    assert matched[0].value == "175"
    assert matched[0].unit == "Nm"


def test_gt_tie_rod_end_nut(extractor):
    """Q5: Verify tie-rod end nut torque from manual page 24."""
    res = extractor.extract("Tie-rod end nut torque")
    assert res.status == "success"
    assert any(p in [24, 59, 585] for p in res.pages_referenced)

    # Ground truth: 115 Nm (85 lb-ft)
    matched = [s for s in res.specifications if "tie-rod end" in s.component.lower()]
    assert len(matched) > 0
    assert matched[0].value == "115"
    assert matched[0].unit == "Nm"


def test_gt_brake_caliper_flow_bolt(extractor):
    """Q6: Verify brake caliper flow bolt torque from manual page 618/636."""
    res = extractor.extract("Brake caliper flow bolt torque")
    assert res.status == "success"

    # Ground truth: 35 Nm (26 lb-ft)
    matched = [s for s in res.specifications if "flow bolt" in s.component.lower()]
    assert len(matched) > 0
    assert matched[0].value == "35"
    assert matched[0].unit == "Nm"
