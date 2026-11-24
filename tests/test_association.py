"""Tests for the box overlap association (standard library only)."""

from ppe_compliance.detection.association import (
    assign_to_persons,
    best_matches,
    boxes_overlap,
    match_infractions,
    overlap_ratio,
    owner_index,
)

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


def test_best_matches_keeps_highest_confidence_per_lower_case_label() -> None:
    detections = [
        {"box": [130, 100, 170, 140], "label": "Hardhat", "conf": 0.4},
        {"box": [130, 100, 170, 140], "label": "HARDHAT", "conf": 0.7},
        {"box": [110, 200, 190, 300], "label": "NO-Safety Vest", "conf": 0.3},
        {"box": [500, 500, 550, 550], "label": "NO-Gloves", "conf": 0.9},  # not associated
    ]
    assert best_matches(PERSON, detections) == {"hardhat": 0.7, "no-safety vest": 0.3}
    assert best_matches(PERSON, []) == {}
    assert best_matches(PERSON, detections, threshold=1.0) == {"hardhat": 0.7, "no-safety vest": 0.3}


def test_zero_threshold_still_needs_real_overlap() -> None:
    assert not boxes_overlap([0, 0, 10, 10], [10, 0, 20, 10], threshold=0.0)
    assert boxes_overlap([0, 0, 10, 10], [9, 9, 20, 20], threshold=0.0)


def test_box_partly_inside_person_uses_small_box_area() -> None:
    # a helmet box sticking out of the top of the person: half of it inside
    helmet = [130, 70, 170, 130]  # 40 x 60, rows 100..130 inside -> 40*30 / 2400 = 0.5
    assert abs(overlap_ratio(helmet, PERSON) - 0.5) < 1e-9


LEFT = [0, 0, 100, 300]
RIGHT = [90, 0, 190, 300]  # overlaps LEFT by a 10 px strip


def test_box_between_two_people_goes_only_to_the_best_one() -> None:
    helmet = [70, 0, 110, 40]  # 30/40 inside LEFT, 20/40 inside RIGHT
    assert owner_index(helmet, [LEFT, RIGHT]) == 0
    assert owner_index(helmet, [RIGHT, LEFT]) == 1
    owned = assign_to_persons([LEFT, RIGHT], [{"box": helmet, "label": "Hardhat", "conf": 0.9}])
    assert len(owned[0]) == 1 and owned[1] == []
    assert best_matches(RIGHT, owned[1]) == {}


def test_missing_class_box_has_a_single_owner() -> None:
    det = {"box": [80, 0, 130, 40], "label": "NO-Hardhat", "conf": 0.5}  # 10 in LEFT, 40 in RIGHT
    owned = assign_to_persons([LEFT, RIGHT], [det])
    assert owned[0] == [] and owned[1] == [det]


def test_equal_ratio_tie_prefers_lower_person_index() -> None:
    mid = [80, 0, 110, 30]  # 20 px inside each person
    assert overlap_ratio(mid, LEFT) == overlap_ratio(mid, RIGHT)
    assert owner_index(mid, [LEFT, RIGHT]) == 0
    assert owner_index(mid, [RIGHT, LEFT]) == 0  # same geometry, first listed wins


def test_equal_ratio_tie_prefers_larger_intersection() -> None:
    # Ratio is intersection / min(area): a small person fully inside the box and a
    # large person fully containing the box both give 1.0; the larger intersection wins.
    box = [0, 0, 100, 100]
    small_person = [20, 20, 40, 40]   # ratio 1.0, intersection 400
    large_person = [0, 0, 300, 300]   # ratio 1.0, intersection 10000
    assert owner_index(box, [small_person, large_person]) == 1
    assert owner_index(box, [large_person, small_person]) == 0


def test_box_below_minimum_overlap_is_unassigned() -> None:
    assert owner_index([500, 500, 550, 550], [LEFT, RIGHT]) is None
    # 4 % of the smaller box: below 0.05 -> nobody; 5 % -> owner
    assert owner_index([0, 0, 100, 100], [[96, 0, 200, 100]]) is None
    assert owner_index([0, 0, 100, 100], [[95, 0, 200, 100]]) == 0
    assert assign_to_persons([LEFT], [{"box": [500, 500, 550, 550]}]) == [[]]


def test_no_persons() -> None:
    assert assign_to_persons([], [{"box": [0, 0, 10, 10]}]) == []
