"""OpenCV drawing and image conversion helpers."""

from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

Color = Tuple[int, int, int]

#: BGR colors: positive PPE / compliant person in green, infractions in red.
GREEN: Color = (0, 255, 0)
RED: Color = (0, 0, 255)


def draw_labeled_box(
    frame: np.ndarray,
    box: Sequence[int],
    text: str,
    color: Color,
    font_scale: float = 0.4,
) -> None:
    """Draw a rectangle with a text label above it, in place.

    Args:
        frame: BGR image to draw on.
        box: ``(x1, y1, x2, y2)``.
        text: Label text.
        color: BGR color.
        font_scale: OpenCV font scale.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1
    )


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
