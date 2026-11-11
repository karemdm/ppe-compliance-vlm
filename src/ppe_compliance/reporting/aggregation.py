"""Per-person consolidation of frame-level infractions (standard library only).

The pipeline feeds it with *confirmed* violations (after the temporal rule of
:mod:`ppe_compliance.tracking.temporal`), so the counts are frames in a confirmed state.
Persons are keyed by the persistent person id of :mod:`ppe_compliance.tracking.reid`
(the tracker id when re-identification is off).
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple


@dataclass
class FrameInfraction:
    """A person associated with infractions in one frame.

    Attributes:
        person_id: Persistent person id (tracker id when re-ID is off).
        missing_ppes: Confirmed infraction labels of the person.
        crop: Person crop (BGR image array); typed as ``Any`` to stay
            independent of numpy.
        track_id: Tracker id of the box, or ``None`` if unknown.
    """

    person_id: int
    missing_ppes: List[str]
    crop: Any = None
    track_id: Optional[int] = None


@dataclass
class PersonRecord:
    """Accumulated statistics for one person.

    Attributes:
        appearances: Number of frames with at least one associated infraction.
        ppe_counts: Frames per infraction label, in first-seen order.
        latest_crop: Crop from the most recent frame with an infraction.
        label_sets: Number of frames per exact set of labels. It lets
            :meth:`PersonInfractionAggregator.select_reports` restrict the
            report to a subset of labels without reprocessing the video.
        label_crops: Per label, ``(appearance number, crop)`` of the most
            recent frame with that label (used for the crop of a filtered report).
        track_ids: Tracker ids of the frames with an infraction, in first-seen order.
    """

    appearances: int = 0
    ppe_counts: Dict[str, int] = field(default_factory=dict)
    latest_crop: Any = None
    label_sets: Counter = field(default_factory=Counter)
    label_crops: Dict[str, Tuple[int, Any]] = field(default_factory=dict)
    track_ids: List[int] = field(default_factory=list)


@dataclass
class PersonReport:
    """Consolidated result for one person that passed the thresholds.

    Attributes:
        person_id: Persistent person id.
        appearances: Frames with at least one infraction.
        infractions: Infraction labels that passed ``min_ratio``.
        ppe_counts: Frame counts of the kept infraction labels.
        crop: Latest person crop.
        track_ids: Tracker ids merged into this person by re-identification.
    """

    person_id: int
    appearances: int
    infractions: List[str]
    ppe_counts: Dict[str, int]
    crop: Any = None
    track_ids: List[int] = field(default_factory=list)


class PersonInfractionAggregator:
    """Accumulates infractions per person over a video."""

    def __init__(self) -> None:
        """Create an empty aggregator."""
        self._records: Dict[int, PersonRecord] = {}

    @property
    def records(self) -> Dict[int, PersonRecord]:
        """Per-person records keyed by person id (first-seen order)."""
        return self._records

    def update(self, item: FrameInfraction) -> None:
        """Add one frame-level infraction.

        Each call counts as one appearance; every label in ``missing_ppes``
        is counted once.

        Args:
            item: The frame infraction to add.
        """
        record = self._records.setdefault(item.person_id, PersonRecord())
        record.appearances += 1
        record.latest_crop = item.crop
        if item.track_id is not None and item.track_id not in record.track_ids:
            record.track_ids.append(item.track_id)
        labels = list(dict.fromkeys(item.missing_ppes))
        for label in labels:
            record.ppe_counts[label] = record.ppe_counts.get(label, 0) + 1
            record.label_crops[label] = (record.appearances, item.crop)
        record.label_sets[frozenset(labels)] += 1

    def update_many(self, items: Iterable[FrameInfraction]) -> None:
        """Add all infractions of a frame.

        Args:
            items: Frame infractions.
        """
        for item in items:
            self.update(item)

    def select_reports(
        self,
        min_appearances: int = 3,
        min_ratio: float = 0.10,
        labels: Optional[Iterable[str]] = None,
    ) -> List[PersonReport]:
        """Select persons and infractions that are consistent over time.

        A person is kept when ``appearances >= min_appearances`` and at least
        one infraction label has ``count / appearances >= min_ratio``.

        When ``labels`` is given (case-insensitive), the statistics are
        recomputed as if only those items had been monitored: a frame counts
        as an appearance only if it contains at least one of them, and the
        crop is the latest one of a frame with one of them. Items are
        confirmed independently of each other, so this gives the same result
        as processing the video with that filter and the UI can change the
        filter without running the models again.

        Args:
            min_appearances: Minimum frames with a (confirmed) infraction.
            min_ratio: Minimum per-label ratio.
            labels: Optional subset of infraction labels to report.

        Returns:
            Reports in first-seen order of the persons.
        """
        wanted: Optional[FrozenSet[str]] = None if labels is None else frozenset(x.lower() for x in labels)
        reports: List[PersonReport] = []
        for person_id, record in self._records.items():
            appearances, counts, crop = _filtered_stats(record, wanted)
            if appearances < min_appearances:
                continue
            kept = {
                label: count
                for label, count in counts.items()
                if count / appearances >= min_ratio
            }
            if not kept:
                continue
            reports.append(
                PersonReport(
                    person_id=person_id,
                    appearances=appearances,
                    infractions=list(kept),
                    ppe_counts=kept,
                    crop=crop,
                    track_ids=list(record.track_ids),
                )
            )
        return reports


def _filtered_stats(
    record: PersonRecord, wanted: Optional[FrozenSet[str]]
) -> Tuple[int, Dict[str, int], Any]:
    """Return ``(appearances, per-label counts, latest crop)`` restricted to ``wanted``.

    Args:
        record: Person record.
        wanted: Lower-case labels to keep, or ``None`` for all labels.

    Returns:
        Appearances, label counts (labels in first-seen order) and the crop of
        the most recent frame with a wanted label (``None`` if there is none).
    """
    if wanted is None:
        return record.appearances, dict(record.ppe_counts), record.latest_crop
    appearances = sum(n for s, n in record.label_sets.items() if any(x.lower() in wanted for x in s))
    counts = {label: c for label, c in record.ppe_counts.items() if label.lower() in wanted}
    latest = [record.label_crops[label] for label in counts]
    crop = max(latest, key=lambda entry: entry[0])[1] if latest else None
    return appearances, counts, crop


def total_infractions(reports: Iterable[PersonReport]) -> int:
    """Sum the number of consolidated infraction labels over all reports.

    Args:
        reports: Selected reports.

    Returns:
        Total count of (person, infraction label) pairs.
    """
    return sum(len(r.infractions) for r in reports)
