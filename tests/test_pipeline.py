"""Tests of process_video with a fake detector and a synthetic video (no inference)."""

import numpy as np
import pytest

pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")

from ppe_compliance import pipeline  # noqa: E402
from ppe_compliance.detection.detector import PersonDetection  # noqa: E402
from ppe_compliance.detection.models import get_model_spec  # noqa: E402
from ppe_compliance.tracking.reid import ReIDConfig  # noqa: E402
from ppe_compliance.tracking.temporal import ItemState, Observation, TemporalConfig  # noqa: E402

FAST = TemporalConfig(window_seconds=0.5, min_frames=2, lost_seconds=1.0)
NO_REID = ReIDConfig(enabled=False)


def make_video(path, n, size=(64, 48), fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for _ in range(n):
        writer.write(np.full((size[1], size[0], 3), 90, np.uint8))
    writer.release()
    return str(path)


class FakeDetector:
    spec = get_model_spec("safetyvision")

    def __init__(self, per_frame):
        self.per_frame = per_frame  # callable(frame_number) -> list of PersonDetection
        self.targets = []

    def process_frame(self, frame, frame_number, targets=None):
        self.targets.append(targets)
        return frame.copy(), self.per_frame(frame_number), [f"--- Frame {frame_number} ---"]


def person(track_id, box, obs):
    return PersonDetection(track_id, list(box), {"no-hardhat": obs},
                           ["NO-Hardhat"] if obs is Observation.NOT_WORN else [])


def test_missing_file_raises_ioerror(tmp_path) -> None:
    with pytest.raises(IOError):
        next(pipeline.process_video(FakeDetector(lambda n: []), str(tmp_path / "none.mp4"), None, FAST, NO_REID))


def test_zero_frames_yield_nothing(tmp_path) -> None:
    video = make_video(tmp_path / "empty.avi", 0)
    try:
        steps = list(pipeline.process_video(FakeDetector(lambda n: []), video, None, FAST, NO_REID))
    except IOError:
        steps = []  # some OpenCV builds refuse to open a file without frames
    assert steps == []


def test_confirmed_violation_and_frame_times(tmp_path) -> None:
    video = make_video(tmp_path / "v.avi", 12, fps=10.0)
    det = FakeDetector(lambda n: [person(5, (4, 4, 40, 44), Observation.NOT_WORN)])
    steps = list(pipeline.process_video(det, video, ["no-hardhat"], FAST, NO_REID))
    assert [s.index for s in steps] == list(range(1, 13))
    assert all(s.total == 12 for s in steps)
    assert det.targets[0] == ["no-hardhat"]
    # window 0.5 s at 10 fps: the state changes at frame 6 (t = 0.5 s), not before
    first = next(s for s in steps if s.infractions)
    assert first.index == 6
    assert first.persons[0].state is ItemState.VIOLATION
    assert first.infractions[0].labels == ["NO-Hardhat"] and first.infractions[0].person_id == 5
    assert first.infractions[0].crop.shape == (48, 54, 3)  # padded by 10 %, grown to 64 px, clipped to the frame
    assert all(s.persons[0].state is ItemState.UNKNOWN for s in steps[:5])
    assert steps[-1].annotated.shape == (48, 64, 3)


def test_duplicate_person_boxes_are_counted_once(tmp_path) -> None:
    video = make_video(tmp_path / "v.avi", 10)
    boxes = lambda n: [person(1, (4, 4, 30, 44), Observation.NOT_WORN),  # noqa: E731
                       person(1, (6, 6, 32, 46), Observation.NOT_WORN)]
    steps = list(pipeline.process_video(FakeDetector(boxes), video, None, FAST, NO_REID))
    confirmed = [s for s in steps if s.infractions]
    assert confirmed and all(len(s.infractions) == 1 for s in confirmed)
    assert all(len(s.persons) == 2 for s in steps)


def test_box_outside_frame_gives_no_crop(tmp_path) -> None:
    video = make_video(tmp_path / "v.avi", 10)
    det = FakeDetector(lambda n: [person(1, (200, 200, 300, 300), Observation.NOT_WORN)])
    steps = list(pipeline.process_video(det, video, None, FAST, NO_REID))
    assert any(s.persons[0].state is ItemState.VIOLATION for s in steps)
    assert all(s.infractions == [] for s in steps)  # empty crop is never reported


def test_reid_pending_then_decided(tmp_path) -> None:
    video = make_video(tmp_path / "v.avi", 4, size=(64, 128))

    class Embedder:
        def embed(self, crops):
            return [[1.0, 0.0] for _ in crops]

    cfg = ReIDConfig(min_samples=2, min_box_height=10)
    det = FakeDetector(lambda n: [person(9, (2, 2, 40, 120), Observation.WORN)])
    steps = list(pipeline.process_video(det, video, None, FAST, cfg, Embedder()))
    assert steps[0].person_ids == [None] and steps[0].persons[0].identified is False
    assert steps[1].person_ids == [1] and len(steps[1].reid_decisions) == 1
    assert "is new person #1" in " ".join(steps[1].logs)
    assert all(s.person_ids == [1] for s in steps[2:])


def test_video_fps_fallback() -> None:
    class Cap:
        def __init__(self, value):
            self.value = value

        def get(self, prop):
            return self.value

    assert pipeline.video_fps(Cap(25.0)) == 25.0
    for bad in (0.0, -1.0, float("nan")):
        assert pipeline.video_fps(Cap(bad)) == pipeline.DEFAULT_FPS


def test_capture_is_released_when_the_consumer_stops_early(tmp_path, monkeypatch) -> None:
    video = make_video(tmp_path / "v.avi", 6)
    released = []
    real_capture = cv2.VideoCapture

    class Spy:
        def __init__(self, path):
            self._cap = real_capture(path)

        def __getattr__(self, name):
            return getattr(self._cap, name)

        def release(self):
            released.append(True)
            self._cap.release()

    monkeypatch.setattr(pipeline.cv2, "VideoCapture", Spy)
    gen = pipeline.process_video(FakeDetector(lambda n: []), video, None, FAST, NO_REID)
    next(gen)
    gen.close()
    assert released == [True]


class _NoCountCapture:
    """Wraps a real capture but reports no frame count."""

    def __init__(self, real) -> None:
        self._real = real

    def get(self, prop):
        return 0.0 if prop == cv2.CAP_PROP_FRAME_COUNT else self._real.get(prop)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_unknown_frame_count_gives_indeterminate_progress(tmp_path, monkeypatch) -> None:
    video = make_video(tmp_path / "v.avi", 4)
    real = cv2.VideoCapture
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda path: _NoCountCapture(real(path)))
    steps = list(pipeline.process_video(FakeDetector(lambda n: []), video, None, FAST, NO_REID))
    assert [s.index for s in steps] == [1, 2, 3, 4]
    assert all(s.total == 0 and s.progress is None for s in steps)


def test_known_frame_count_gives_progress(tmp_path) -> None:
    video = make_video(tmp_path / "v.avi", 4)
    steps = list(pipeline.process_video(FakeDetector(lambda n: []), video, None, FAST, NO_REID))
    assert [s.progress for s in steps] == [0.25, 0.5, 0.75, 1.0]
