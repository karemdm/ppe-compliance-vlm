"""Frame-by-frame video processing shared by the Streamlit app and the CLI."""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import cv2
import numpy as np

from .aggregation import FrameInfraction
from .detector import PPEDetector


@dataclass
class FrameStep:
    """Result of processing one video frame.

    Attributes:
        index: 1-based frame index.
        total: Total frame count reported by the container (at least 1).
        annotated: Annotated BGR frame.
        infractions: Per-person infractions of this frame.
        logs: Log lines of this frame.
    """

    index: int
    total: int
    annotated: np.ndarray
    infractions: List[FrameInfraction]
    logs: List[str]


def process_video(
    detector: PPEDetector,
    video_path: str,
    target_infractions: Optional[Iterable[str]] = None,
) -> Iterator[FrameStep]:
    """Run the detector on every frame of a video.

    Args:
        detector: Detector instance (call ``reset_tracking`` before a new video).
        video_path: Path to the video file.
        target_infractions: Infraction labels to monitor (all when ``None``).

    Yields:
        One :class:`FrameStep` per decoded frame.

    Raises:
        IOError: If the video cannot be opened.
    """
    targets = None if target_infractions is None else list(target_infractions)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            annotated, infractions, logs = detector.process_frame(frame, index, targets)
            yield FrameStep(index, max(total, index), annotated, infractions, logs)
    finally:
        cap.release()
