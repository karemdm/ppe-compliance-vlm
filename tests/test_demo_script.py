"""Smoke tests of the pure helpers of scripts/make_demo_video.py (no models, no ffmpeg)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_video.py"
spec = importlib.util.spec_from_file_location("make_demo_video", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_ascii_text_and_wrap() -> None:
    assert demo.ascii_text("a — b × c…") == "a - b x c..."
    assert demo.ascii_text("é") == "?"
    lines = demo.wrap("one two three four five six seven", 120, 0.6, 1)
    assert len(lines) > 1 and " ".join(lines) == "one two three four five six seven"
    assert demo.wrap("", 100, 0.6, 1) == [""]


def test_card_and_freeze_frame_shapes() -> None:
    img = demo.card((640, 360), "Title", [("line", 0.9, demo.LIGHT)])
    assert img.shape == (360, 640, 3)
    base = np.full((360, 640, 3), 100, np.uint8)
    crop = np.full((120, 40, 3), 50, np.uint8)
    info = {"number": 1, "total": 3, "track_id": 2, "violations": ["NO-Hardhat"], "model": "m"}
    # box partly outside the frame must not raise
    out = demo.freeze_frame(base, [-10, 20, 100, 340], crop, info, [("Analyzing", 1.0, demo.WHITE)])
    assert out.shape == base.shape


def test_parser_defaults() -> None:
    args = demo.build_parser().parse_args([])
    assert args.max_vlm_checks == 3 and args.min_crop_height == 150
