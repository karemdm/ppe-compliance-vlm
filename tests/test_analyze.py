"""Tests of the headless analysis with a fake detector and fake pipeline (no inference)."""

import json

import numpy as np
import pytest

pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")

from ppe_compliance.reporting.aggregation import FrameInfraction  # noqa: E402
from ppe_compliance.pipeline import FrameStep  # noqa: E402
from ppe_compliance.tracking.reid import ReIDDecision  # noqa: E402
from ppe_compliance.ui import analyze  # noqa: E402


def make_video(path, n=3, size=(64, 48), fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for _ in range(n):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()
    return str(path)


def fake_steps(n, labels=("NO-Hardhat",), person_id=4):
    frame = np.zeros((48, 64, 3), np.uint8)
    for i in range(1, n + 1):
        decisions = [ReIDDecision(10, person_id, False)] if i == 1 else []
        if i == 2:
            decisions = [ReIDDecision(11, person_id, True, 0.9)]
        yield FrameStep(i, n, frame.copy(), [FrameInfraction(person_id, list(labels), frame, track_id=10 + (i > 1))],
                        [], [], [person_id], decisions)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(analyze, "PPEDetector", lambda cfg: object())
    return monkeypatch


def test_report_json(patched, tmp_path) -> None:
    patched.setattr(analyze, "process_video", lambda *a, **k: fake_steps(4))
    video = make_video(tmp_path / "in.avi")
    out = tmp_path / "sub" / "report.json"
    assert analyze.main(["--input", video, "--output", str(out), "--no-vlm"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["frames_processed"] == 4
    assert report["persons_with_infractions"] == 1 and report["total_consolidated_infractions"] == 1
    (person,) = report["persons"]
    assert person["person_id"] == 4 and person["track_ids"] == [10, 11]
    assert person["appearances"] == 4 and person["infraction_counts"] == {"NO-Hardhat": 4}
    assert person["description_source"] == "template"
    assert report["reid_summary"] == {"tracks_assigned": 2, "tracks_reidentified": 1, "persons": 1}
    assert report["settings"]["temporal"]["min_frames"] == 5


def test_no_reid_summary_is_null_and_zero_frames(patched, tmp_path) -> None:
    patched.setattr(analyze, "process_video", lambda *a, **k: iter(()))
    video = make_video(tmp_path / "in.avi")
    out = tmp_path / "r.json"
    save = tmp_path / "annotated.mp4"
    assert analyze.main(["--input", video, "--output", str(out), "--no-vlm", "--no-reid",
                         "--save-video", str(save)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["frames_processed"] == 0 and report["persons"] == []
    assert report["reid_summary"] is None
    assert not save.exists()  # no frame decoded: no video written


def test_save_video_matches_frame_size_and_fps(patched, tmp_path) -> None:
    patched.setattr(analyze, "process_video", lambda *a, **k: fake_steps(3))
    video = make_video(tmp_path / "in.avi", fps=10.0)
    save = tmp_path / "o" / "annotated.mp4"
    assert analyze.main(["--input", video, "--output", str(tmp_path / "r.json"), "--no-vlm",
                         "--save-video", str(save)]) == 0
    cap = cv2.VideoCapture(str(save))
    assert cap.isOpened()
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 10.0) < 0.5
    cap.release()


def test_unreadable_video_returns_1(patched, tmp_path) -> None:
    def broken(*a, **k):
        raise IOError("Cannot open video: x")
        yield  # pragma: no cover

    patched.setattr(analyze, "process_video", broken)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert analyze.main(["--input", str(bad), "--output", str(tmp_path / "r.json"), "--no-vlm"]) == 1
    assert not (tmp_path / "r.json").exists()


def test_detector_failure_returns_1(monkeypatch, tmp_path) -> None:
    def boom(cfg):
        raise FileNotFoundError("PPE weights missing")

    monkeypatch.setattr(analyze, "PPEDetector", boom)
    video = make_video(tmp_path / "in.avi")
    assert analyze.main(["--input", video, "--output", str(tmp_path / "r.json"), "--no-vlm"]) == 1


def test_argument_errors_exit_2(tmp_path) -> None:
    for argv in (["--input", str(tmp_path / "missing.mp4"), "--output", "o.json"],
                 ["--input", "x", "--output", "o.json", "--ppe-model", "chd", "--infractions", "no-gloves"],
                 ["--input", "x", "--output", "o.json", "--min-majority", "0.3"]):
        with pytest.raises(SystemExit) as exc:
            analyze.main(argv)
        assert exc.value.code == 2, argv
