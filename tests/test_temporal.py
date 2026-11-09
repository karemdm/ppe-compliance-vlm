"""Tests for the temporal confirmation rule (standard library only)."""

from ppe_compliance.association import best_matches, observe_items
from ppe_compliance.models import get_model_spec
from ppe_compliance.temporal import (
    ItemState,
    ItemTracker,
    Observation,
    PersonComplianceTracker,
    TemporalConfig,
    overall_state,
    resolve_observation,
)

FPS = 10.0
W, N, U = Observation.WORN, Observation.NOT_WORN, Observation.UNKNOWN
CFG = TemporalConfig(window_seconds=3.0, min_majority=0.6, min_coverage=0.3, lost_seconds=2.0)


def _run(tracker: ItemTracker, observations: list, start_frame: int = 0) -> list:
    """Feed one observation per frame at FPS and return the state after each frame."""
    return [tracker.update((start_frame + i) / FPS, obs) for i, obs in enumerate(observations)]


def test_violation_only_after_window() -> None:
    tracker = ItemTracker(CFG)
    states = _run(tracker, [N] * 40)
    # t = i / 10; the window (3.0 s) is reached at frame 30.
    assert all(s is ItemState.UNKNOWN for s in states[:30])
    assert all(s is ItemState.VIOLATION for s in states[30:])


def test_short_noise_burst_does_not_flip() -> None:
    for burst in (1, 2, 3):
        tracker = ItemTracker(CFG)
        _run(tracker, [W] * 40)
        assert tracker.state is ItemState.OK
        states = _run(tracker, [N] * burst + [W] * 20, start_frame=40)
        assert all(s is ItemState.OK for s in states)


def test_unknown_frames_do_not_flip() -> None:
    tracker = ItemTracker(CFG)
    _run(tracker, [N] * 40)
    assert tracker.state is ItemState.VIOLATION
    states = _run(tracker, [U] * 60, start_frame=40)
    assert all(s is ItemState.VIOLATION for s in states)


def test_low_coverage_keeps_state() -> None:
    tracker = ItemTracker(CFG)
    _run(tracker, [N] * 40)
    # 1 informative frame in 10 (coverage 0.1 < 0.3), all of them worn. While the
    # old NOT_WORN frames leave the window, coverage >= 0.3 and a WORN majority
    # never hold at the same time, so the state is kept.
    states = _run(tracker, ([W] + [U] * 9) * 12, start_frame=40)
    assert all(s is ItemState.VIOLATION for s in states)


def test_low_coverage_never_confirms() -> None:
    tracker = ItemTracker(CFG)
    states = _run(tracker, ([N] + [U] * 4) * 12)
    assert all(s is ItemState.UNKNOWN for s in states)


def test_returns_to_ok_only_after_worn_majority() -> None:
    tracker = ItemTracker(CFG)
    _run(tracker, [N] * 40)
    states = _run(tracker, [W] * 40, start_frame=40)
    # Window has 31 entries; WORN needs >= 0.6 of them: 19 worn frames.
    first_ok = next(i for i, s in enumerate(states) if s is ItemState.OK)
    assert first_ok == 18
    assert all(s is ItemState.VIOLATION for s in states[:first_ok])
    assert all(s is ItemState.OK for s in states[first_ok:])


def test_mixed_window_keeps_state() -> None:
    tracker = ItemTracker(CFG)
    _run(tracker, [W] * 40)
    states = _run(tracker, [W, N] * 30, start_frame=40)  # 50/50: no side reaches 0.6
    assert all(s is ItemState.OK for s in states)


def test_eviction_keeps_counts_consistent() -> None:
    tracker = ItemTracker(CFG)
    pattern = [W, N, U, N, W, W, U]
    observations = pattern * 20
    for i, obs in enumerate(observations):
        tracker.update(i / FPS, obs)
        window = observations[max(0, i - 30): i + 1]  # entries with t >= t_now - 3.0
        assert tracker.total == len(window)
        assert tracker.worn == window.count(W)
        assert tracker.not_worn == window.count(N)


def test_person_tracker_prune_after_lost_seconds() -> None:
    tracker = PersonComplianceTracker(CFG)
    tracker.update(0.0, 1, {"no-hardhat": N})
    tracker.update(0.0, 2, {"no-hardhat": N})
    tracker.update(1.5, 2, {"no-hardhat": N})
    assert tracker.prune(2.0) == []
    assert tracker.prune(2.1) == [1]
    assert tracker.track_ids == [2]
    assert tracker.prune(3.6) == [2]
    assert tracker.track_ids == []


def test_person_tracker_resets_after_long_gap() -> None:
    tracker = PersonComplianceTracker(CFG, labels={"no-hardhat": "NO-Hardhat"})
    for i in range(40):
        status = tracker.update(i / FPS, 7, {"no-hardhat": N})
    assert status.state is ItemState.VIOLATION
    assert status.confirmed_violations == ["NO-Hardhat"]
    status = tracker.update(3.9 + 2.5, 7, {"no-hardhat": N})  # gap > lost_seconds
    assert status.state is ItemState.UNKNOWN


def test_overall_state() -> None:
    v, ok, u = ItemState.VIOLATION, ItemState.OK, ItemState.UNKNOWN
    assert overall_state({}) is u
    assert overall_state({"a": u, "b": u}) is u
    assert overall_state({"a": ok, "b": u}) is ok
    assert overall_state({"a": ok, "b": v, "c": u}) is v


def test_person_status_lists_confirmed_violations() -> None:
    labels = {"no-hardhat": "NO-Hardhat", "no-mask": "NO-Mask", "fall-detected": "Fall-Detected"}
    tracker = PersonComplianceTracker(CFG, labels)
    for i in range(35):
        status = tracker.update(i / FPS, 3, {"no-hardhat": N, "no-mask": W, "fall-detected": U})
    assert status.items == {"no-hardhat": ItemState.VIOLATION, "no-mask": ItemState.OK,
                            "fall-detected": ItemState.UNKNOWN}
    assert status.confirmed_violations == ["NO-Hardhat"]
    assert status.state is ItemState.VIOLATION


def test_resolve_observation_both_classes() -> None:
    assert resolve_observation(None, None) is U
    assert resolve_observation(0.4, None) is W
    assert resolve_observation(None, 0.2) is N
    assert resolve_observation(0.7, 0.3) is W
    assert resolve_observation(0.3, 0.7) is N
    assert resolve_observation(0.5, 0.5) is N  # tie: conservative


def test_observe_items_from_boxes() -> None:
    items = get_model_spec("safetyvision").monitored_items()
    person = [100, 100, 200, 400]
    detections = [
        {"box": [120, 100, 160, 130], "label": "Hardhat", "conf": 0.8},
        {"box": [125, 100, 165, 130], "label": "NO-Hardhat", "conf": 0.3},
        {"box": [110, 150, 190, 250], "label": "NO-Safety Vest", "conf": 0.5},
        {"box": [110, 150, 190, 250], "label": "NO-Safety Vest", "conf": 0.6},
        {"box": [500, 500, 520, 520], "label": "NO-Mask", "conf": 0.9},  # not on the person
    ]
    matches = best_matches(person, detections)
    assert matches == {"hardhat": 0.8, "no-hardhat": 0.3, "no-safety vest": 0.6}
    obs = observe_items(matches, items)
    assert obs["no-hardhat"] is W
    assert obs["no-safety vest"] is N
    assert obs["no-mask"] is U
    assert obs["fall-detected"] is U


def test_monitored_items_registry() -> None:
    items = get_model_spec("safetyvision").monitored_items()
    assert [i.key for i in items] == list(get_model_spec("safetyvision").infraction_labels())
    by_key = {i.key: i for i in items}
    assert (by_key["no-hardhat"].label, by_key["no-hardhat"].positive) == ("NO-Hardhat", "hardhat")
    assert by_key["no_harness"].positive is None
    assert by_key["fall-detected"].positive is None


def test_config_validation() -> None:
    for kwargs in ({"min_majority": 0.5}, {"min_coverage": 1.5}, {"window_seconds": -1.0}):
        try:
            TemporalConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted {kwargs}")


def test_min_frames_blocks_decision_on_few_informative_frames() -> None:
    cfg = TemporalConfig(window_seconds=1.0, min_majority=0.6, min_coverage=0.0, lost_seconds=2.0, min_frames=5)
    item = ItemTracker(cfg)
    for i in range(30):  # 30 fps, 1 s: only 3 informative frames, the rest unknown
        obs = Observation.NOT_WORN if i in (25, 27, 29) else Observation.UNKNOWN
        state = item.update(i / 30.0, obs)
    assert state is ItemState.UNKNOWN
    for i in range(30, 33):
        state = item.update(i / 30.0, Observation.NOT_WORN)
    assert state is ItemState.VIOLATION
