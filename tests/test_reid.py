"""Tests for the re-identification of persons across tracker ids (standard library only)."""

import pytest

from ppe_compliance.aggregation import FrameInfraction, PersonInfractionAggregator
from ppe_compliance.reid import (
    ReIDConfig,
    ReIdentifier,
    cosine,
    max_overlap_with_others,
    mean_vector,
    normalize,
    usable_for_embedding,
)

FPS = 10.0
RED_SHIRT = [1.0, 0.0, 0.0, 0.1]
BLUE_SHIRT = [0.0, 1.0, 0.0, 0.1]
GREEN_SHIRT = [0.0, 0.0, 1.0, 0.1]
CFG = ReIDConfig(threshold=0.85, margin=0.03, min_samples=3, max_pending_seconds=0.5)


def _jitter(vector: list, k: int) -> list:
    """Small deterministic variation of an embedding."""
    return [v + 0.02 * ((k + i) % 3 - 1) for i, v in enumerate(vector)]


def _run(reid: ReIdentifier, track_id: int, vector: list, start: int, frames: int) -> list:
    """Feed one track for ``frames`` frames and return the person id after each frame."""
    return [reid.update((start + i) / FPS, track_id, _jitter(vector, i))[0] for i in range(frames)]


def test_vector_helpers() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 0]) == 0.0
    assert normalize([3, 4]) == pytest.approx([0.6, 0.8])
    assert mean_vector([[1, 0], [0, 1]]) == pytest.approx([2 ** -0.5, 2 ** -0.5])
    with pytest.raises(ValueError):
        cosine([1], [1, 2])


def test_pending_until_min_samples() -> None:
    reid = ReIdentifier(CFG)
    ids = _run(reid, 5, RED_SHIRT, 0, 4)
    assert ids == [None, None, 1, 1]  # decided at the third embedding
    assert reid.person_of(5) == 1


def test_person_reidentified_after_leaving() -> None:
    reid = ReIdentifier(CFG)
    _run(reid, 1, RED_SHIRT, 0, 30)          # person A, track 1, leaves at t = 2.9 s
    _run(reid, 2, BLUE_SHIRT, 0, 80)         # person B stays in the scene
    ids = _run(reid, 7, RED_SHIRT, 60, 10)  # A re-enters with a new track id at t = 6 s
    assert ids[2:] == [1] * 8
    assert reid.identities[1].track_ids == [1, 7]
    assert len(reid.identities) == 2


def test_different_person_gets_new_id() -> None:
    reid = ReIdentifier(CFG)
    _run(reid, 1, RED_SHIRT, 0, 30)
    ids = _run(reid, 2, GREEN_SHIRT, 40, 5)
    assert ids[-1] == 2


def test_simultaneous_identity_is_not_a_candidate() -> None:
    reid = ReIdentifier(CFG)
    _run(reid, 1, RED_SHIRT, 0, 10)
    # A second person in the same red shirt appears while track 1 is still visible.
    for i in range(10, 20):
        reid.update(i / FPS, 1, RED_SHIRT)
        pid, _ = reid.update(i / FPS, 2, RED_SHIRT)
    assert pid == 2


def test_ambiguous_match_creates_new_identity() -> None:
    reid = ReIdentifier(ReIDConfig(threshold=0.5, margin=0.2, min_samples=1))
    reid.update(0.0, 1, [1.0, 0.0])
    reid.update(0.0, 2, [0.0, 1.0])
    pid, decision = reid.update(1.0, 3, [1.0, 0.9])  # about equally close to both
    assert pid == 3 and not decision.reused


def test_max_absence_limits_candidates() -> None:
    reid = ReIdentifier(ReIDConfig(min_samples=1, max_absence_seconds=5.0))
    reid.update(0.0, 1, RED_SHIRT)
    assert reid.update(4.0, 2, RED_SHIRT)[0] == 1
    assert reid.update(20.0, 3, RED_SHIRT)[0] == 2


def test_no_embedding_decides_new_after_max_pending() -> None:
    reid = ReIdentifier(CFG)
    _run(reid, 1, RED_SHIRT, 0, 10)
    ids = [reid.update(3.0 + i / FPS, 9, None)[0] for i in range(7)]
    assert ids[:5] == [None] * 5   # waits max_pending_seconds (0.5 s)
    assert ids[5] == 2             # then a new person, since nothing can be compared


def test_disabled_returns_track_id() -> None:
    reid = ReIdentifier(ReIDConfig(enabled=False))
    assert reid.update(0.0, 42, RED_SHIRT) == (42, None)
    assert not reid.needs_embedding(0.0, 42)


def test_needs_embedding_sampling() -> None:
    reid = ReIdentifier(ReIDConfig(min_samples=1, sample_seconds=0.5))
    assert reid.needs_embedding(0.0, 1)  # unknown track
    reid.update(0.0, 1, RED_SHIRT)
    assert not reid.needs_embedding(0.2, 1)
    assert reid.needs_embedding(0.5, 1)


def test_gallery_is_bounded() -> None:
    reid = ReIdentifier(ReIDConfig(min_samples=1, gallery_size=4, sample_seconds=0.0))
    for i in range(20):
        reid.update(i / FPS, 1, _jitter(RED_SHIRT, i))
    assert len(reid.identities[1].gallery) == 4


def test_usable_for_embedding() -> None:
    cfg = ReIDConfig(min_box_height=64, max_overlap=0.3)
    box = [0, 0, 50, 100]
    assert usable_for_embedding(box, [], cfg)
    assert not usable_for_embedding([0, 0, 50, 60], [], cfg)        # too small
    assert not usable_for_embedding(box, [[0, 0, 50, 50]], cfg)    # half occluded
    assert usable_for_embedding(box, [[40, 0, 90, 100]], cfg)      # 20% overlap
    assert max_overlap_with_others(box, [[100, 100, 120, 120]]) == 0.0


def test_config_validation() -> None:
    for kwargs in ({"threshold": 1.5}, {"margin": -0.1}, {"min_samples": 0},
                   {"gallery_size": 0}, {"max_overlap": 2.0}, {"max_pending_seconds": -1}):
        with pytest.raises(ValueError):
            ReIDConfig(**kwargs)


def test_aggregator_merges_tracks_of_one_person() -> None:
    agg = PersonInfractionAggregator()
    for track_id in (3, 3, 8, 8, 8):
        agg.update(FrameInfraction(1, ["NO-Hardhat"], crop=f"c{track_id}", track_id=track_id))
    (report,) = agg.select_reports(min_appearances=3)
    assert report.person_id == 1
    assert report.track_ids == [3, 8]
    assert report.appearances == 5
