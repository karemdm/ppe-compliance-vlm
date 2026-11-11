"""Tests for configuration helpers (standard library only)."""

from ppe_compliance.config import DetectorConfig, ReportConfig, resolve_tracker


def test_defaults() -> None:
    d, r = DetectorConfig(), ReportConfig()
    assert (d.person_conf, d.person_iou, d.person_classes) == (0.45, 0.20, (0,))
    assert (d.ppe_conf, d.ppe_iou, d.overlap_threshold) == (0.15, 0.20, 0.05)
    assert d.tracker == "bytetrack.yaml"
    assert d.person_weights == "weights/yolov8n.pt"
    assert d.ppe_model == "safetyvision"
    assert (r.min_appearances, r.min_ratio) == (3, 0.10)


def test_resolve_tracker() -> None:
    assert resolve_tracker("bytetrack.yaml") == "bytetrack.yaml"
    assert resolve_tracker("configs/bytetrack_custom.yaml").endswith("bytetrack_custom.yaml")
    assert resolve_tracker("configs/botsort_reid.yaml").endswith("botsort_reid.yaml")


def test_cli_builds_temporal_config() -> None:
    import argparse

    from ppe_compliance.ui.cli import add_pipeline_args, build_configs
    from ppe_compliance.tracking.temporal import TemporalConfig

    parser = argparse.ArgumentParser()
    add_pipeline_args(parser)
    *_, temporal = build_configs(parser.parse_args([]))
    assert temporal == TemporalConfig()
    *_, temporal = build_configs(parser.parse_args(["--confirm-seconds", "1.5", "--min-majority", "0.8"]))
    assert (temporal.window_seconds, temporal.min_majority) == (1.5, 0.8)
