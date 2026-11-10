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


def _as_tuples(reports: list) -> list:
    return [(r.person_id, r.appearances, r.infractions, r.ppe_counts) for r in reports]


def test_label_filter_matches_processing_with_filter():
    frames = [
        ["no-hardhat"], ["no-hardhat", "no-gloves"], ["no-gloves"], ["no-gloves"],
        ["no-gloves"], ["no-hardhat"], ["no-gloves"], ["no-gloves"], ["no-gloves"], ["no-gloves"],
    ]
    wanted = {"no-hardhat"}
    # Processing once with every label, then filtering the report...
    full = PersonInfractionAggregator()
    _feed(full, 1, frames)
    filtered = full.select_reports(3, 0.10, labels=wanted)
    # ...must equal processing the video with only the wanted labels monitored.
    direct = PersonInfractionAggregator()
    _feed(direct, 1, [[l for l in f if l in wanted] for f in frames if set(f) & wanted])
    assert _as_tuples(filtered) == _as_tuples(direct.select_reports(3, 0.10))
    assert filtered[0].appearances == 3


def test_label_filter_none_keeps_previous_behaviour():
    agg = PersonInfractionAggregator()
    _feed(agg, 7, [["no-mask"], ["no-mask"], ["no-mask"]])
    assert _as_tuples(agg.select_reports(3, 0.10)) == _as_tuples(agg.select_reports(3, 0.10, labels=None))
    assert agg.select_reports(3, 0.10, labels=[]) == []


def test_label_filter_is_case_insensitive():
    agg = PersonInfractionAggregator()
    _feed(agg, 1, [["NO-Hardhat", "NO-Mask"], ["NO-Hardhat"], ["NO-Hardhat"], ["NO-Mask"]])
    (report,) = agg.select_reports(3, 0.10, labels=["no-hardhat"])
    assert report.infractions == ["NO-Hardhat"]
    assert report.appearances == 3


def test_label_filter_uses_latest_crop_of_wanted_labels():
    agg = PersonInfractionAggregator()
    _feed(agg, 1, [["NO-Hardhat"], ["NO-Hardhat"], ["NO-Hardhat", "NO-Mask"], ["NO-Mask"], ["NO-Mask"]])
    (report,) = agg.select_reports(3, 0.10, labels=["no-hardhat"])
    assert report.crop == "crop-1-2"  # last frame with NO-Hardhat, not the last frame overall
    (report,) = agg.select_reports(3, 0.10)
    assert report.crop == "crop-1-4"
