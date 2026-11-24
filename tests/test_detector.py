"""Tests of PPEDetector.process_frame with stubbed models (no inference)."""

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from ppe_compliance.config import DetectorConfig  # noqa: E402
from ppe_compliance.detection.detector import PPEDetector  # noqa: E402
from ppe_compliance.detection.models import get_model_spec  # noqa: E402
from ppe_compliance.tracking.temporal import Observation  # noqa: E402


def make_detector(persons, ppe, model="safetyvision", **cfg) -> PPEDetector:
    """Build a detector without loading weights; stub the two model calls."""
    det = PPEDetector.__new__(PPEDetector)
    det.config = DetectorConfig(ppe_model=model, **cfg)
    det.spec = get_model_spec(model)
    det._infraction_classes = det.spec.infraction_classes()
    det._positive_classes = det.spec.positive_classes()
    det._items = det.spec.monitored_items()
    det._track_persons = lambda frame: persons
    det._detect_ppe = lambda frame: ppe
    return det


FRAME = np.zeros((480, 640, 3), np.uint8)
PERSON = ([100, 50, 220, 400], 7)


def test_worn_missing_and_unknown_items() -> None:
    ppe = [
        {"box": [130, 50, 190, 100], "label": "Hardhat", "conf": 0.8},
        {"box": [110, 150, 210, 300], "label": "NO-Safety Vest", "conf": 0.6},
        {"box": [500, 400, 560, 460], "label": "NO-Gloves", "conf": 0.9},  # not on the person
    ]
    _, dets, logs = make_detector([PERSON], ppe).process_frame(FRAME, 1)
    (d,) = dets
    assert d.track_id == 7
    assert d.observations["no-hardhat"] is Observation.WORN
    assert d.observations["no-safety vest"] is Observation.NOT_WORN
    assert d.observations["no-gloves"] is Observation.UNKNOWN
    assert d.raw_infractions == ["NO-Safety Vest"]
    assert any("Raw infractions: 2" in line for line in logs)


def test_targets_filter_items_and_are_case_insensitive() -> None:
    ppe = [{"box": [110, 150, 210, 300], "label": "NO-Safety Vest", "conf": 0.6},
           {"box": [130, 50, 190, 100], "label": "NO-Hardhat", "conf": 0.7}]
    detector = make_detector([PERSON], ppe)
    _, (d,), _ = detector.process_frame(FRAME, 1, ["NO-Hardhat"])
    assert list(d.observations) == ["no-hardhat"]
    assert d.raw_infractions == ["NO-Hardhat"]
    _, (d,), _ = detector.process_frame(FRAME, 1, ["no-hardhat"])
    assert list(d.observations) == ["no-hardhat"]
    _, (d,), _ = detector.process_frame(FRAME, 1, [])
    assert d.observations == {} and d.raw_infractions == []


def test_tie_between_worn_and_missing_counts_as_missing() -> None:
    ppe = [{"box": [130, 50, 190, 100], "label": "Hardhat", "conf": 0.5},
           {"box": [130, 50, 190, 100], "label": "NO-Hardhat", "conf": 0.5}]
    _, (d,), _ = make_detector([PERSON], ppe).process_frame(FRAME, 1, ["no-hardhat"])
    assert d.observations["no-hardhat"] is Observation.NOT_WORN


def test_ignored_classes_and_no_persons() -> None:
    ppe = [{"box": [130, 50, 190, 100], "label": "Person", "conf": 0.9},
           {"box": [130, 50, 190, 100], "label": "Safety Cone", "conf": 0.9}]
    annotated, dets, logs = make_detector([], ppe, model="chd").process_frame(FRAME, 3)
    assert dets == []
    assert not annotated.any()  # ignored classes are not drawn
    assert annotated is not FRAME
    assert logs[0] == "--- Frame 3 ---"


def test_unsupported_item_is_not_observed_for_chd() -> None:
    ppe = [{"box": [110, 150, 210, 300], "label": "NO-Gloves", "conf": 0.9}]
    _, (d,), _ = make_detector([PERSON], ppe, model="chd").process_frame(FRAME, 1)
    assert "no-gloves" not in d.observations


def test_overlap_threshold_is_used() -> None:
    # hardhat box with ~2% of its area inside the person box
    ppe = [{"box": [218, 50, 318, 150], "label": "Hardhat", "conf": 0.9}]
    strict = make_detector([PERSON], ppe, overlap_threshold=0.05)
    loose = make_detector([PERSON], ppe, overlap_threshold=0.01)
    assert strict.process_frame(FRAME, 1)[1][0].observations["no-hardhat"] is Observation.UNKNOWN
    assert loose.process_frame(FRAME, 1)[1][0].observations["no-hardhat"] is Observation.WORN
