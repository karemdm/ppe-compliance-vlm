"""Person tracking + PPE detection and per-person observations for a single frame."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from .association import best_matches, observe_items
from ..config import DetectorConfig, resolve_tracker, resolve_weights
from ..ui.drawing import GREEN, RED, draw_labeled_box
from .models import get_model_spec, resolve_model_weights, unmatched_class_names
from ..tracking.temporal import Observation

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """A tracked person in one frame with its per-item observations.

    Attributes:
        track_id: Tracker id.
        box: ``(x1, y1, x2, y2)`` person box.
        observations: Observation per monitored item key (e.g. ``"no-hardhat"``).
        raw_infractions: Infraction labels observed in this frame (before
            temporal confirmation), for logging.
    """

    track_id: int
    box: List[int]
    observations: Dict[str, Observation]
    raw_infractions: List[str]


class PPEDetector:
    """Tracks persons and associates PPE / infraction detections with them.

    Two models are used: a COCO YOLO model tracked with ByteTrack for persons
    and one of the bundled PPE YOLO models (see :mod:`ppe_compliance.detection.models`).
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """Load both models.

        Args:
            config: Detector settings; defaults are used when ``None``.

        Raises:
            ValueError: If the PPE model key is unknown.
            FileNotFoundError: If the PPE weights or tracker file are missing.
        """
        self.config = config or DetectorConfig()
        self._tracker = resolve_tracker(self.config.tracker)
        self.spec = get_model_spec(self.config.ppe_model)
        self._infraction_classes = self.spec.infraction_classes()
        self._positive_classes = self.spec.positive_classes()
        self._items = self.spec.monitored_items()
        ppe_path = resolve_model_weights(self.spec)
        logger.info("Loading PPE model '%s' from %s", self.spec.key, ppe_path)
        self.ppe_model = YOLO(ppe_path)
        missing, unexpected = unmatched_class_names(self.spec, self.ppe_model.names.values())
        if missing or unexpected:
            logger.warning("PPE model '%s' class names differ from the registry: missing=%s, unexpected=%s",
                           self.spec.key, missing, unexpected)
        self._person_weights = resolve_weights(self.config.person_weights)
        self.person_model = YOLO(self._person_weights)

    def reset_tracking(self, tracker: Optional[str] = None) -> None:
        """Start a fresh tracking session (new track ids), e.g. for a new video.

        Args:
            tracker: Optional new tracker name or YAML path.
        """
        if tracker is not None:
            self.config.tracker = tracker
            self._tracker = resolve_tracker(tracker)
        self.person_model = YOLO(self._person_weights)

    def _track_persons(self, frame: np.ndarray) -> List[Tuple[List[int], int]]:
        cfg = self.config
        result = self.person_model.track(
            frame,
            conf=cfg.person_conf,
            iou=cfg.person_iou,
            classes=list(cfg.person_classes),
            tracker=self._tracker,
            persist=True,
            verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        boxes = result.boxes.xyxy.int().cpu().tolist()
        ids = result.boxes.id.int().cpu().tolist()
        return list(zip(boxes, ids))

    def _detect_ppe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        result = self.ppe_model(
            frame,
            conf=self.config.ppe_conf,
            iou=self.config.ppe_iou,
            imgsz=self.spec.imgsz,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        return [
            {"box": b, "label": self.ppe_model.names[c], "conf": float(s)}
            for b, c, s in zip(boxes, classes, confs)
        ]

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        target_infractions: Optional[Iterable[str]] = None,
    ) -> Tuple[np.ndarray, List[PersonDetection], List[str]]:
        """Run tracking, PPE detection and association on one frame.

        Positive PPE boxes are drawn in green and monitored infraction boxes in
        red (thin lines). Person boxes are not drawn here: their color depends
        on the temporal rule applied by :func:`pipeline.process_video`.

        Args:
            frame: BGR frame.
            frame_number: 1-based frame index (used in the logs).
            target_infractions: Infraction labels to monitor; all infractions
                supported by the PPE model when ``None``. Only these items are
                observed.

        Returns:
            The annotated frame, one :class:`PersonDetection` per tracked
            person and log lines.
        """
        targets = set(self._infraction_classes if target_infractions is None else target_infractions)
        items = [item for item in self._items if item.key in targets]
        annotated = frame.copy()
        logs = [f"--- Frame {frame_number} ---"]

        persons = self._track_persons(frame)
        relevant: List[Dict[str, Any]] = []
        n_infractions = 0
        for det in self._detect_ppe(frame):
            label_lower = det["label"].lower()
            is_infraction = label_lower in self._infraction_classes and label_lower in targets
            if not (is_infraction or label_lower in self._positive_classes):
                continue
            n_infractions += is_infraction
            draw_labeled_box(annotated, det["box"], f"{det['label']} ({det['conf']:.2f})",
                             RED if is_infraction else GREEN, thickness=1)
            relevant.append(det)

        logs.append(f"Tracked persons: {len(persons)} | Raw infractions: {n_infractions}")

        detections: List[PersonDetection] = []
        for box, person_id in persons:
            matches = best_matches(box, relevant, self.config.overlap_threshold)
            observations = observe_items(matches, items)
            raw = [i.label for i in items if observations[i.key] is Observation.NOT_WORN]
            if raw:
                logs.append(f"Person #{person_id} raw infractions {raw}")
            detections.append(PersonDetection(person_id, box, observations, raw))
        return annotated, detections, logs
