"""Association of PPE / infraction boxes with person boxes (standard library only).

Boxes are ``(x1, y1, x2, y2)`` sequences in pixel coordinates.
"""

from typing import Dict, Iterable, Mapping, Sequence, Set

from .models import MonitoredItem
from .temporal import Observation, resolve_observation

Box = Sequence[float]


def box_area(box: Box) -> float:
    """Return the area of a box (0 for degenerate boxes).

    Args:
        box: ``(x1, y1, x2, y2)``.

    Returns:
        The box area in pixels.
    """
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap_ratio(box1: Box, box2: Box) -> float:
    """Compute intersection area divided by the smaller box area.

    Unlike IoU, this ratio reaches 1.0 when a small box (e.g. a missing
    helmet detection) lies entirely inside a large one (the person).

    Args:
        box1: First box.
        box2: Second box.

    Returns:
        ``intersection / min(area1, area2)``, or 0.0 if there is no overlap.
    """
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    min_area = min(box_area(box1), box_area(box2))
    if min_area <= 0:
        return 0.0
    return (inter_w * inter_h) / min_area


def boxes_overlap(box1: Box, box2: Box, threshold: float = 0.05) -> bool:
    """Tell whether two boxes overlap enough to be associated.

    Args:
        box1: First box.
        box2: Second box.
        threshold: Minimum :func:`overlap_ratio`.

    Returns:
        True if the ratio is at least ``threshold``.
    """
    ratio = overlap_ratio(box1, box2)
    return ratio > 0.0 and ratio >= threshold


def match_infractions(
    person_box: Box,
    infractions: Iterable[Mapping[str, object]],
    threshold: float = 0.05,
) -> Set[str]:
    """Collect the labels of the infraction boxes associated with a person.

    Args:
        person_box: Person bounding box.
        infractions: Items with ``"box"`` and ``"label"`` keys.
        threshold: Minimum :func:`overlap_ratio`.

    Returns:
        The set of associated infraction labels.
    """
    return {
        str(inf["label"])
        for inf in infractions
        if boxes_overlap(inf["box"], person_box, threshold)  # type: ignore[arg-type]
    }


def best_matches(
    person_box: Box,
    detections: Iterable[Mapping[str, object]],
    threshold: float = 0.05,
) -> Dict[str, float]:
    """Return the highest confidence per lower-case label among associated boxes.

    Args:
        person_box: Person bounding box.
        detections: Items with ``"box"``, ``"label"`` and ``"conf"`` keys.
        threshold: Minimum :func:`overlap_ratio`.

    Returns:
        Mapping of lower-case label to the best confidence.
    """
    best: Dict[str, float] = {}
    for det in detections:
        if boxes_overlap(det["box"], person_box, threshold):  # type: ignore[arg-type]
            label = str(det["label"]).lower()
            conf = float(det["conf"])  # type: ignore[arg-type]
            if conf > best.get(label, -1.0):
                best[label] = conf
    return best


def observe_items(matches: Mapping[str, float], items: Iterable[MonitoredItem]) -> Dict[str, Observation]:
    """Turn the associated labels of a person into one observation per item.

    Args:
        matches: Output of :func:`best_matches`.
        items: Items to observe.

    Returns:
        Observation per item key (see :func:`temporal.resolve_observation`).
    """
    return {
        item.key: resolve_observation(
            matches.get(item.positive) if item.positive else None, matches.get(item.key)
        )
        for item in items
    }
