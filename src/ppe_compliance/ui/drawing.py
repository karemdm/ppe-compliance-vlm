"""OpenCV drawing and image conversion helpers."""

from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ..tracking.temporal import ItemState, PersonStatus

Color = Tuple[int, int, int]

#: BGR colors: positive PPE / compliant person in green, infractions in red.
GREEN: Color = (0, 255, 0)
RED: Color = (0, 0, 255)
#: BGR color of a person whose state is not confirmed yet.
GRAY: Color = (160, 160, 160)

STATE_COLORS = {ItemState.UNKNOWN: GRAY, ItemState.OK: GREEN, ItemState.VIOLATION: RED}


def draw_labeled_box(
    frame: np.ndarray,
    box: Sequence[int],
    text: str,
    color: Color,
    font_scale: float = 0.4,
    thickness: int = 2,
) -> None:
    """Draw a rectangle with a text label above it, in place.

    Args:
        frame: BGR image to draw on.
        box: ``(x1, y1, x2, y2)``.
        text: Label text.
        color: BGR color.
        font_scale: OpenCV font scale.
        thickness: Rectangle line thickness.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1
    )


def draw_person_status(frame: np.ndarray, box: Sequence[int], status: PersonStatus) -> None:
    """Draw a person box colored by its confirmed state, in place.

    Gray with ``#id checking`` while unknown, green with ``#id OK``, red with
    the confirmed violation labels. ``#id`` is the person id (the tracker id
    when re-identification is off).

    Args:
        frame: BGR image to draw on.
        box: ``(x1, y1, x2, y2)`` person box.
        status: Confirmed status from the temporal rule.
    """
    if status.state is ItemState.VIOLATION:
        text = f"#{status.track_id} " + ", ".join(status.confirmed_violations)
    elif status.state is ItemState.OK:
        text = f"#{status.track_id} OK"
    else:
        text = f"#{status.track_id} checking"
    draw_labeled_box(frame, box, text, STATE_COLORS[status.state], font_scale=0.5, thickness=2)


def draw_pending_person(frame: np.ndarray, box: Sequence[int], track_id: int) -> None:
    """Draw a new track whose re-identification is pending (gray, thin), in place.

    Args:
        frame: BGR image to draw on.
        box: ``(x1, y1, x2, y2)`` person box.
        track_id: Tracker id.
    """
    draw_labeled_box(frame, box, f"track {track_id} re-id...", GRAY, font_scale=0.5, thickness=1)


def crop_box(frame: np.ndarray, box: Sequence[int]) -> np.ndarray:
    """Crop a box from a frame, clipped to the image bounds.

    Args:
        frame: BGR image.
        box: ``(x1, y1, x2, y2)``.

    Returns:
        A copy of the cropped region (may be empty).
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    return frame[y1:y2, x1:x2].copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB.

    Args:
        image: BGR image.

    Returns:
        RGB image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR image to a PIL RGB image.

    Args:
        image: BGR image.

    Returns:
        PIL image in RGB mode.
    """
    return Image.fromarray(bgr_to_rgb(image))
