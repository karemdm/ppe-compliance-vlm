"""Tests of crop / drawing helpers and the histogram embedder (synthetic images, no models)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")

from ppe_compliance.tracking.appearance import HistogramEmbedder, build_embedder  # noqa: E402
from ppe_compliance.tracking.reid import cosine  # noqa: E402
from ppe_compliance.tracking.temporal import ItemState, PersonStatus  # noqa: E402
from ppe_compliance.ui.drawing import (  # noqa: E402
    GRAY, GREEN, RED, bgr_to_pil, crop_box, draw_labeled_box, draw_pending_person, draw_person_status,
)


def test_crop_box_clips_to_image() -> None:
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    assert crop_box(frame, [10, 20, 50, 60]).shape == (40, 40, 3)
    assert crop_box(frame, [-30, -30, 20, 10]).shape == (10, 20, 3)
    assert crop_box(frame, [150, 80, 400, 300]).shape == (20, 50, 3)
    assert crop_box(frame, [300, 300, 400, 400]).size == 0
    assert crop_box(frame, [50, 50, 50, 90]).size == 0  # zero width
    assert crop_box(frame, [60, 60, 20, 20]).size == 0  # inverted box
    crop = crop_box(frame, [0, 0, 5, 5])
    crop[:] = 0
    assert frame[:5, :5].any()  # a copy, not a view


def test_crop_box_float_coordinates() -> None:
    frame = np.zeros((50, 50, 3), np.uint8)
    assert crop_box(frame, [1.9, 2.2, 10.7, 12.9]).shape == (10, 9, 3)


def test_bgr_to_pil_swaps_channels() -> None:
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0] = 255  # blue in BGR
    pixel = bgr_to_pil(img).getpixel((0, 0))
    assert pixel == (0, 0, 255)


def test_draw_status_colors_and_in_place() -> None:
    for state, color in ((ItemState.VIOLATION, RED), (ItemState.OK, GREEN), (ItemState.UNKNOWN, GRAY)):
        frame = np.zeros((100, 100, 3), np.uint8)
        draw_person_status(frame, [20, 30, 80, 90], PersonStatus(3, state, {}, ["NO-Hardhat"]))
        assert tuple(frame[60, 20]) == color  # left edge of the box
    frame = np.zeros((100, 100, 3), np.uint8)
    draw_pending_person(frame, [20, 30, 80, 90], 4)
    assert tuple(frame[60, 20]) == GRAY


def test_draw_box_at_image_border_does_not_fail() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    draw_labeled_box(frame, [-10, -10, 60, 60], "x", RED)
    draw_labeled_box(frame, [5, 0, 5, 0], "degenerate", RED)


def solid(color, size=(40, 80)):
    return np.full((size[1], size[0], 3), color, np.uint8)


def test_histogram_embedder_unit_norm_and_identity() -> None:
    emb = HistogramEmbedder()
    a, same, none = emb.embed([solid((200, 30, 30)), solid((200, 30, 30)), np.zeros((0, 10, 3), np.uint8)])
    assert none is None
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-5
    assert cosine(a, same) > 0.999


def test_histogram_embedder_separates_colors_and_is_scale_invariant() -> None:
    emb = HistogramEmbedder()
    red, blue, red_big = emb.embed([solid((0, 0, 255)), solid((255, 0, 0)), solid((0, 0, 255), (160, 320))])
    assert cosine(red, red_big) > 0.999
    assert cosine(red, blue) < 0.5


def test_histogram_embedder_tiny_and_none_crops() -> None:
    emb = HistogramEmbedder()
    (tiny,) = emb.embed([solid((10, 200, 10), (1, 1))])
    assert tiny is not None and len(tiny) == 6 * (16 * 4 + 8)
    assert emb.embed([None]) == [None]


def test_build_embedder_histogram_is_case_insensitive() -> None:
    assert isinstance(build_embedder("Histogram"), HistogramEmbedder)
