"""Frame-by-frame video processing shared by the Streamlit app and the CLI."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

import cv2
import numpy as np

from .aggregation import FrameInfraction
from .appearance import build_embedder
from .detector import PPEDetector
from .drawing import crop_box, draw_pending_person, draw_person_status
from .reid import ReIDConfig, ReIDDecision, ReIdentifier, usable_for_embedding
from .temporal import ItemState, PersonComplianceTracker, PersonStatus, TemporalConfig

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
        persons: Confirmed status of every tracked person in this frame, in
            the order of the detector output. ``PersonStatus.track_id`` holds
            the persistent person id (the tracker id when re-ID is off); a
            track still being re-identified has ``identified=False`` and its
            tracker id.
        logs: Log lines of this frame.
        person_ids: Person id per tracked person (``None`` while pending).
        reid_decisions: Re-ID decisions taken in this frame.
    """

    index: int
    total: int
    annotated: np.ndarray
    infractions: List[FrameInfraction]
    persons: List[PersonStatus]
    logs: List[str]
    person_ids: List[Optional[int]] = field(default_factory=list)
    reid_decisions: List[ReIDDecision] = field(default_factory=list)


def video_fps(cap: "cv2.VideoCapture") -> float:
    """Return the frame rate of an opened capture, or :data:`DEFAULT_FPS`.

    Args:
        cap: Opened video capture.

    Returns:
        Frames per second (> 0).
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 0 else DEFAULT_FPS


def _describe(decision: ReIDDecision) -> str:
    sim = "" if decision.similarity is None else f" (similarity {decision.similarity:.2f})"
    if decision.reused:
        return f"Track {decision.track_id} re-identified as person #{decision.person_id}{sim}"
    return f"Track {decision.track_id} is new person #{decision.person_id}" + (
        f" (best similarity {decision.similarity:.2f})" if decision.similarity is not None else "")


def process_video(
    detector: PPEDetector,
    video_path: str,
    target_infractions: Optional[Iterable[str]] = None,
    temporal: Optional[TemporalConfig] = None,
    reid: Optional[ReIDConfig] = None,
    embedder: Optional[Any] = None,
) -> Iterator[FrameStep]:
    """Run the detector, re-ID and the temporal confirmation on every frame of a video.

    The frame time is ``(index - 1) / fps``. Tracker ids are mapped to
    persistent person ids by :class:`ReIdentifier` (unless re-ID is disabled),
    and the temporal rule and the reports use the person id. A person box is
    drawn gray, green or red from its confirmed state, and only confirmed
    violations are returned in :attr:`FrameStep.infractions`.

    Args:
        detector: Detector instance (call ``reset_tracking`` before a new video).
        video_path: Path to the video file.
        target_infractions: Infraction labels to monitor (all when ``None``).
        temporal: Temporal rule parameters (defaults when ``None``).
        reid: Re-ID parameters (defaults, i.e. enabled, when ``None``).
        embedder: Appearance embedder to reuse (built from ``reid.embedder``
            when ``None``).

    Yields:
        One :class:`FrameStep` per decoded frame.

    Raises:
        IOError: If the video cannot be opened.
    """
    targets = None if target_infractions is None else list(target_infractions)
    labels = {item.key: item.label for item in detector.spec.monitored_items()}
    compliance = PersonComplianceTracker(temporal, labels)
    reid_cfg = reid or ReIDConfig()
    reidentifier = ReIdentifier(reid_cfg)
    if reid_cfg.enabled and embedder is None:
        embedder = build_embedder(reid_cfg.embedder)
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

            embeddings: List[Optional[List[float]]] = [None] * len(detections)
            if reid_cfg.enabled and detections:
                boxes = [d.box for d in detections]
                wanted = [
                    i for i, d in enumerate(detections)
                    if reidentifier.needs_embedding(t, d.track_id)
                    and usable_for_embedding(d.box, boxes[:i] + boxes[i + 1:], reid_cfg)
                ]
                if wanted:
                    vectors = embedder.embed([crop_box(frame, detections[i].box) for i in wanted])
                    for i, vector in zip(wanted, vectors):
                        embeddings[i] = vector

            infractions: List[FrameInfraction] = []
            statuses: List[PersonStatus] = []
            person_ids: List[Optional[int]] = []
            decisions: List[ReIDDecision] = []
            done: Dict[int, PersonStatus] = {}
            for det, vector in zip(detections, embeddings):
                person_id, decision = reidentifier.update(t, det.track_id, vector)
                if decision is not None:
                    decisions.append(decision)
                    logs.append(_describe(decision))
                person_ids.append(person_id)
                if person_id is None:
                    status = PersonStatus(det.track_id, ItemState.UNKNOWN, identified=False)
                    statuses.append(status)
                    draw_pending_person(annotated, det.box, det.track_id)
                    continue
                if person_id in done:  # two boxes of one person in a frame: count it once
                    statuses.append(done[person_id])
                    draw_person_status(annotated, det.box, done[person_id])
                    continue
                status = done[person_id] = compliance.update(t, person_id, det.observations)
                statuses.append(status)
                draw_person_status(annotated, det.box, status)
                if status.confirmed_violations:
                    logs.append(f"Person #{person_id} (track {det.track_id}) confirmed "
                                f"{status.confirmed_violations}")
                    crop = crop_box(frame, det.box)
                    if crop.size > 0:
                        infractions.append(FrameInfraction(person_id, status.confirmed_violations, crop,
                                                           track_id=det.track_id))
            compliance.prune(t)
            yield FrameStep(index, max(total, index), annotated, infractions, statuses, logs,
                            person_ids, decisions)
    finally:
        cap.release()
