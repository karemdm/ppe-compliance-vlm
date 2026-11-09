"""Person tracking + PPE detection for a single frame."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from .aggregation import FrameInfraction
from .association import match_infractions
from .config import DetectorConfig, resolve_tracker, resolve_weights
from .models import get_model_spec, resolve_model_weights, unmatched_class_names
from .drawing import GREEN, RED, crop_box, draw_labeled_box

logger = logging.getLogger(__name__)


class PPEDetector:
    """Tracks persons and associates PPE infraction detections with them.

    Two models are used: a COCO YOLO model tracked with ByteTrack for persons
    and one of the bundled PPE YOLO models (see :mod:`ppe_compliance.models`).
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
    ) -> Tuple[np.ndarray, List[FrameInfraction], List[str]]:
        """Run tracking, PPE detection and association on one frame.

        Positive PPE boxes are drawn in green and monitored infraction boxes in
        red. Persons are drawn red when associated with an infraction.

        Args:
            frame: BGR frame.
            frame_number: 1-based frame index (used in the logs).
            target_infractions: Infraction labels to monitor; all infractions
                supported by the PPE model when ``None``.

        Returns:
            The annotated frame, the per-person infractions and log lines.
        """
        targets = set(self._infraction_classes if target_infractions is None else target_infractions)
        annotated = frame.copy()
        logs = [f"--- Frame {frame_number} ---"]

        persons = self._track_persons(frame)
        infractions_raw: List[Dict[str, Any]] = []
        for det in self._detect_ppe(frame):
            label_lower = det["label"].lower()
            is_infraction = label_lower in self._infraction_classes and label_lower in targets
            if not (is_infraction or label_lower in self._positive_classes):
                continue
            color = RED if is_infraction else GREEN
            draw_labeled_box(annotated, det["box"], f"{det['label']} ({det['conf']:.2f})", color)
            if is_infraction:
                infractions_raw.append(det)

        logs.append(f"Tracked persons: {len(persons)} | Raw infractions: {len(infractions_raw)}")

        frame_infractions: List[FrameInfraction] = []
        for box, person_id in persons:
            missing = match_infractions(box, infractions_raw, self.config.overlap_threshold)
            color = RED if missing else GREEN
            draw_labeled_box(annotated, box, f"Person #{person_id}", color, font_scale=0.5)
            if not missing:
                continue
            labels = sorted(missing)
            logs.append(f"Person #{person_id} matched with {labels}")
            crop = crop_box(frame, box)
            if crop.size > 0:
                frame_infractions.append(FrameInfraction(person_id, labels, crop))

        return annotated, frame_infractions, logs
