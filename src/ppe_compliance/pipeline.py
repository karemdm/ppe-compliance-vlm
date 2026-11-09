"""Frame-by-frame video processing shared by the Streamlit app and the CLI."""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import cv2
import numpy as np

from .aggregation import FrameInfraction
from .detector import PPEDetector
from .drawing import crop_box, draw_person_status
from .temporal import PersonComplianceTracker, PersonStatus, TemporalConfig

#: Frame rate assumed when the container does not report one.
DEFAULT_FPS = 30.0


@dataclass
class FrameStep:
    """Result of processing one video frame.

    Attributes:
        index: 1-based frame index.
        total: Total frame count reported by the container (at least 1).
        annotated: Annotated BGR frame.
        infractions: Per-person *confirmed* infractions of this frame.
        persons: Confirmed status of every tracked person in this frame.
        logs: Log lines of this frame.
    """

    index: int
    total: int
    annotated: np.ndarray
    infractions: List[FrameInfraction]
    persons: List[PersonStatus]
    logs: List[str]


def video_fps(cap: "cv2.VideoCapture") -> float:
    """Return the frame rate of an opened capture, or :data:`DEFAULT_FPS`.

    Args:
        cap: Opened video capture.

    Returns:
        Frames per second (> 0).
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 0 else DEFAULT_FPS


def process_video(
    detector: PPEDetector,
    video_path: str,
    target_infractions: Optional[Iterable[str]] = None,
    temporal: Optional[TemporalConfig] = None,
) -> Iterator[FrameStep]:
    """Run the detector and the temporal confirmation on every frame of a video.

    The frame time is ``(index - 1) / fps``. A person box is drawn gray, green
    or red from its confirmed state, and only confirmed violations are
    returned in :attr:`FrameStep.infractions`.

    Args:
        detector: Detector instance (call ``reset_tracking`` before a new video).
        video_path: Path to the video file.
        target_infractions: Infraction labels to monitor (all when ``None``).
        temporal: Temporal rule parameters (defaults when ``None``).

    Yields:
        One :class:`FrameStep` per decoded frame.

    Raises:
        IOError: If the video cannot be opened.
    """
    targets = None if target_infractions is None else list(target_infractions)
    labels = {item.key: item.label for item in detector.spec.monitored_items()}
    compliance = PersonComplianceTracker(temporal, labels)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = video_fps(cap)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            t = (index - 1) / fps
            annotated, detections, logs = detector.process_frame(frame, index, targets)
            infractions: List[FrameInfraction] = []
            statuses: List[PersonStatus] = []
            for det in detections:
                status = compliance.update(t, det.track_id, det.observations)
                statuses.append(status)
                draw_person_status(annotated, det.box, status)
                if status.confirmed_violations:
                    logs.append(f"Person #{det.track_id} confirmed {status.confirmed_violations}")
                    crop = crop_box(frame, det.box)
                    if crop.size > 0:
                        infractions.append(FrameInfraction(det.track_id, status.confirmed_violations, crop))
            compliance.prune(t)
            yield FrameStep(index, max(total, index), annotated, infractions, statuses, logs)
    finally:
        cap.release()
