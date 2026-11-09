"""Per-person consolidation of frame-level infractions (standard library only)."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class FrameInfraction:
    """A tracked person associated with infractions in one frame.

    Attributes:
        person_id: Tracker id of the person.
        missing_ppes: Infraction labels associated with the person.
        crop: Person crop (BGR image array); typed as ``Any`` to stay
            independent of numpy.
    """

    person_id: int
    missing_ppes: List[str]
    crop: Any = None


@dataclass
class PersonRecord:
    """Accumulated statistics for one tracked person.

    Attributes:
        appearances: Number of frames with at least one associated infraction.
        ppe_counts: Frames per infraction label, in first-seen order.
        latest_crop: Crop from the most recent frame with an infraction.
    """

    appearances: int = 0
    ppe_counts: Dict[str, int] = field(default_factory=dict)
    latest_crop: Any = None


@dataclass
class PersonReport:
    """Consolidated result for one person that passed the thresholds.

    Attributes:
        person_id: Tracker id.
        appearances: Frames with at least one infraction.
        infractions: Infraction labels that passed ``min_ratio``.
        ppe_counts: Frame counts of the kept infraction labels.
        crop: Latest person crop.
    """

    person_id: int
    appearances: int
    infractions: List[str]
    ppe_counts: Dict[str, int]
    crop: Any = None


class PersonInfractionAggregator:
    """Accumulates infractions per tracked person over a video."""

    def __init__(self) -> None:
        """Create an empty aggregator."""
        self._records: Dict[int, PersonRecord] = {}

    @property
    def records(self) -> Dict[int, PersonRecord]:
        """Per-person records keyed by tracker id (first-seen order)."""
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
        for label in dict.fromkeys(item.missing_ppes):
            record.ppe_counts[label] = record.ppe_counts.get(label, 0) + 1

    def update_many(self, items: Iterable[FrameInfraction]) -> None:
        """Add all infractions of a frame.

        Args:
            items: Frame infractions.
        """
        for item in items:
            self.update(item)

    def select_reports(
        self, min_appearances: int = 3, min_ratio: float = 0.10
    ) -> List[PersonReport]:
        """Select persons and infractions that are consistent over time.

        A person is kept when ``appearances >= min_appearances`` and at least
        one infraction label has ``count / appearances >= min_ratio``.

        Args:
            min_appearances: Minimum frames with an infraction.
            min_ratio: Minimum per-label ratio.

        Returns:
            Reports in first-seen order of the persons.
        """
        reports: List[PersonReport] = []
        for person_id, record in self._records.items():
            if record.appearances < min_appearances:
                continue
            kept = {
                label: count
                for label, count in record.ppe_counts.items()
                if count / record.appearances >= min_ratio
            }
            if not kept:
                continue
            reports.append(
                PersonReport(
                    person_id=person_id,
                    appearances=record.appearances,
                    infractions=list(kept),
                    ppe_counts=kept,
                    crop=record.latest_crop,
                )
            )
        return reports


def total_infractions(reports: Iterable[PersonReport]) -> int:
    """Sum the number of consolidated infraction labels over all reports.

    Args:
        reports: Selected reports.

    Returns:
        Total count of (person, infraction label) pairs.
    """
    return sum(len(r.infractions) for r in reports)
