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


def test_resolve_weights_uses_repo_root(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from ppe_compliance import config

    repo = tmp_path / "repo"
    (repo / "weights").mkdir(parents=True)
    (repo / "weights" / "m.pt").write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(config, "_REPO_ROOT", repo)
    monkeypatch.chdir(elsewhere)
    assert config.resolve_weights("weights/m.pt") == str(repo / "weights" / "m.pt")
    # a file in the working directory wins
    (elsewhere / "weights").mkdir()
    (elsewhere / "weights" / "m.pt").write_bytes(b"y")
    assert config.resolve_weights("weights/m.pt") == str(Path("weights/m.pt"))
    # unknown file: Ultralytics auto-download name
    assert config.resolve_weights("weights/other.pt") == "other.pt"


def test_detector_and_report_config_validation() -> None:
    import pytest

    for bad in ({"person_conf": 1.5}, {"ppe_conf": -0.1}, {"ppe_iou": 2}, {"overlap_threshold": -1}):
        with pytest.raises(ValueError):
            DetectorConfig(**bad)
    for bad in ({"min_appearances": 0}, {"min_ratio": 1.5}, {"min_ratio": -0.1}):
        with pytest.raises(ValueError):
            ReportConfig(**bad)
    DetectorConfig(person_conf=0.0, ppe_conf=1.0)  # boundaries are valid


def test_app_main_rejects_bad_values_before_launching(monkeypatch) -> None:
    import pytest

    from ppe_compliance.ui import cli

    launched = []
    monkeypatch.setattr(cli, "launch_app", lambda args: launched.append(args) or 0)
    for bad in (["--min-majority", "0.4"], ["--reid-threshold", "2"], ["--min-ratio", "3"],
                ["--person-conf", "1.2"], ["--min-frames", "0"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(bad)
        assert exc.value.code == 2, bad
    assert launched == []
    assert cli.main(["app", "--no-vlm"]) == 0
    assert launched == [["--no-vlm"]]
