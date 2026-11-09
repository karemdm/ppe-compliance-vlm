"""Tests for per-person aggregation and report thresholds (standard library only)."""

from ppe_compliance.aggregation import (
    FrameInfraction,
    PersonInfractionAggregator,
    total_infractions,
)


def _feed(agg: PersonInfractionAggregator, person_id: int, labels_per_frame: list) -> None:
    for i, labels in enumerate(labels_per_frame):
        agg.update(FrameInfraction(person_id, labels, crop=f"crop-{person_id}-{i}"))


def test_counts_and_latest_crop() -> None:
    agg = PersonInfractionAggregator()
    _feed(agg, 1, [["no-hardhat"], ["no-hardhat", "no-mask"], ["no-mask", "no-mask"]])
    rec = agg.records[1]
    assert rec.appearances == 3
    assert rec.ppe_counts == {"no-hardhat": 2, "no-mask": 2}  # duplicates counted once
    assert rec.latest_crop == "crop-1-2"


def test_min_appearances() -> None:
    agg = PersonInfractionAggregator()
    _feed(agg, 1, [["no-hardhat"]] * 2)
    _feed(agg, 2, [["no-hardhat"]] * 3)
    reports = agg.select_reports(min_appearances=3, min_ratio=0.10)
    assert [r.person_id for r in reports] == [2]


def test_min_ratio_filters_labels() -> None:
    agg = PersonInfractionAggregator()
    # 20 appearances: no-hardhat in all, no-gloves in 2 (0.10, kept), no-mask in 1 (0.05, dropped)
    frames = [["no-hardhat"] for _ in range(20)]
    frames[0] = ["no-hardhat", "no-gloves", "no-mask"]
    frames[1] = ["no-hardhat", "no-gloves"]
    _feed(agg, 7, frames)
    (report,) = agg.select_reports(min_appearances=3, min_ratio=0.10)
    assert report.infractions == ["no-hardhat", "no-gloves"]
    assert report.ppe_counts == {"no-hardhat": 20, "no-gloves": 2}
    assert report.appearances == 20
    assert report.crop == "crop-7-19"


def test_person_dropped_when_no_label_passes_ratio() -> None:
    agg = PersonInfractionAggregator()
    _feed(agg, 3, [["no-hardhat"]] * 3)
    assert agg.select_reports(min_appearances=3, min_ratio=1.01) == []


def test_total_infractions_and_order() -> None:
    agg = PersonInfractionAggregator()
    _feed(agg, 5, [["no-mask", "no-gloves"]] * 3)
    _feed(agg, 2, [["no-hardhat"]] * 4)
    reports = agg.select_reports()
    assert [r.person_id for r in reports] == [5, 2]  # first-seen order
    assert total_infractions(reports) == 3


def test_update_many() -> None:
    agg = PersonInfractionAggregator()
    agg.update_many([FrameInfraction(1, ["no-mask"]), FrameInfraction(2, ["no-mask"])])
    assert set(agg.records) == {1, 2}
