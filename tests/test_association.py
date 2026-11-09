"""Tests for the box overlap association (standard library only)."""

from ppe_compliance.association import boxes_overlap, match_infractions, overlap_ratio

PERSON = [100, 100, 200, 400]


def test_no_overlap_is_zero() -> None:
    assert overlap_ratio([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert not boxes_overlap([0, 0, 10, 10], [20, 20, 30, 30])


def test_touching_edges_do_not_overlap() -> None:
    assert overlap_ratio([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_small_box_inside_person_is_full_ratio() -> None:
    helmet = [130, 100, 170, 140]
    assert overlap_ratio(helmet, PERSON) == 1.0
    assert overlap_ratio(PERSON, helmet) == 1.0  # symmetric


def test_ratio_uses_min_area() -> None:
    # 10x10 box, half of it inside a large box -> 0.5
    assert abs(overlap_ratio([0, 0, 10, 10], [5, 0, 100, 100]) - 0.5) < 1e-9


def test_threshold_boundary() -> None:
    small = [0, 0, 100, 100]  # area 10000
    exact = [95, 0, 200, 100]  # intersection 5x100 = 500 -> ratio 0.05
    below = [96, 0, 200, 100]  # intersection 400 -> 0.04
    assert boxes_overlap(small, exact, threshold=0.05)
    assert not boxes_overlap(small, below, threshold=0.05)


def test_degenerate_box() -> None:
    assert overlap_ratio([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0


def test_match_infractions_collects_labels() -> None:
    infractions = [
        {"box": [130, 100, 170, 140], "label": "no-hardhat"},
        {"box": [110, 200, 190, 300], "label": "no-safety vest"},
        {"box": [500, 500, 550, 550], "label": "no-gloves"},
    ]
    assert match_infractions(PERSON, infractions) == {"no-hardhat", "no-safety vest"}
    assert match_infractions(PERSON, []) == set()
