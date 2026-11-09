"""Configuration defaults and path helpers.

PPE class mappings live in :mod:`ppe_compliance.models`.

Standard library only, so it can be imported by tests and lightweight code.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .models import DEFAULT_PPE_MODEL

logger = logging.getLogger(__name__)

#: Trackers that ship with Ultralytics and are resolved by name.
BUILTIN_TRACKERS: Tuple[str, ...] = ("bytetrack.yaml", "botsort.yaml")

#: Custom tracker configuration bundled with the repository.
CUSTOM_TRACKER = "configs/bytetrack_custom.yaml"

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DetectorConfig:
    """Settings of the person tracker and the PPE detector.

    Attributes:
        person_weights: Path to the person detection weights (COCO YOLO).
        person_conf: Confidence threshold for person detections.
        person_iou: NMS IoU threshold for person detections.
        person_classes: COCO class ids tracked as persons.
        tracker: Ultralytics tracker name or path to a tracker YAML.
        ppe_model: Key of the bundled PPE model (see ``models.PPE_MODELS``).
        ppe_conf: Confidence threshold for PPE detections. Kept low because
            the temporal rule (:mod:`ppe_compliance.temporal`) filters noise.
        ppe_iou: NMS IoU threshold for PPE detections.
        overlap_threshold: Minimum intersection / min(area) to associate a
            PPE or infraction box with a person box.
    """

    person_weights: str = "weights/yolov8n.pt"
    person_conf: float = 0.45
    person_iou: float = 0.20
    person_classes: Tuple[int, ...] = (0,)
    tracker: str = "bytetrack.yaml"
    ppe_model: str = DEFAULT_PPE_MODEL
    ppe_conf: float = 0.15
    ppe_iou: float = 0.20
    overlap_threshold: float = 0.05


@dataclass
class ReportConfig:
    """Thresholds used to consolidate per-person infractions.

    Both thresholds apply to frames with a *confirmed* violation (after the
    temporal rule), not to raw detections.

    Attributes:
        min_appearances: Minimum number of frames in which a person must have
            at least one confirmed violation.
        min_ratio: Minimum ratio (confirmed count / appearances) for a PPE
            infraction to be kept in the report.
    """

    min_appearances: int = 3
    min_ratio: float = 0.10


@dataclass
class VLMConfig:
    """Settings of the vision-language model used for the report text.

    Attributes:
        model_id: Hugging Face model id.
        max_new_tokens: Maximum number of generated tokens.
        repetition_penalty: Repetition penalty for greedy decoding.
    """

    model_id: str = "HuggingFaceM4/idefics2-8b"
    max_new_tokens: int = 60
    repetition_penalty: float = 1.5


def resolve_weights(path: str) -> str:
    """Return a usable weights path.

    If ``path`` does not exist, its file name is returned instead so that
    Ultralytics can auto-download official weights (e.g. ``yolov8n.pt``).

    Args:
        path: Configured weights path.

    Returns:
        The existing path, or its file name as an Ultralytics model name.
    """
    if Path(path).is_file():
        return path
    name = Path(path).name
    logger.warning("Weights '%s' not found, falling back to '%s' (auto-download).", path, name)
    return name


def resolve_tracker(tracker: str) -> str:
    """Resolve a tracker name or YAML path.

    Built-in Ultralytics names are returned unchanged. Relative paths are
    looked up in the current directory first and then in the repository root.

    Args:
        tracker: Tracker name or path.

    Returns:
        A value accepted by the Ultralytics ``track(tracker=...)`` argument.

    Raises:
        FileNotFoundError: If a custom tracker file cannot be found.
    """
    if tracker in BUILTIN_TRACKERS:
        return tracker
    for candidate in (Path(tracker), _REPO_ROOT / tracker):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Tracker configuration not found: {tracker}")


def available_trackers() -> List[str]:
    """List the tracker options offered in the UI.

    Returns:
        ``bytetrack.yaml`` plus the bundled custom config when it is found.
    """
    options = ["bytetrack.yaml"]
    try:
        resolve_tracker(CUSTOM_TRACKER)
        options.append(CUSTOM_TRACKER)
    except FileNotFoundError:
        pass
    return options
