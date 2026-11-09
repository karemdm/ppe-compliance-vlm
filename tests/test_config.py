"""Tests for configuration helpers (standard library only)."""

from ppe_compliance.config import DetectorConfig, ReportConfig, resolve_tracker


def test_defaults_match_prototype() -> None:
    d, r = DetectorConfig(), ReportConfig()
    assert (d.person_conf, d.person_iou, d.person_classes) == (0.45, 0.20, (0,))
    assert (d.ppe_conf, d.ppe_iou, d.overlap_threshold) == (0.25, 0.20, 0.05)
    assert d.tracker == "bytetrack.yaml"
    assert d.person_weights == "weights/yolov8n.pt"
    assert d.ppe_model == "safetyvision"
    assert (r.min_appearances, r.min_ratio) == (3, 0.10)


def test_resolve_tracker() -> None:
    assert resolve_tracker("bytetrack.yaml") == "bytetrack.yaml"
    assert resolve_tracker("configs/bytetrack_custom.yaml").endswith("bytetrack_custom.yaml")
