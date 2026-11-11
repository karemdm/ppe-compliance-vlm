"""Render a demo video: PPE detection, temporal confirmation and a VLM double-check.

The script runs the normal pipeline (:func:`ppe_compliance.pipeline.process_video`)
frame by frame. When a tracked person is confirmed as a violation for the first
time (up to ``--max-vlm-checks`` persons), playback freezes: the frame is
darkened, the person is highlighted and a side panel shows the enlarged crop,
the confirmed violations and an elapsed-time counter while the VLM really runs
on the crop. The VLM answer is then shown for a few seconds and playback
resumes. A title card and an end card are added, the raw video is re-encoded
with ffmpeg (H.264) and every VLM check is logged to JSON.

The wait shown in the video is shortened to ``--max-wait-seconds`` of video
by sampling the counter; the counter always shows the real elapsed time.

Usage (from the repository root)::

    PYTHONPATH=src python scripts/make_demo_video.py --input outputs/demo_input.mp4 \\
        --vlm-model HuggingFaceTB/SmolVLM-500M-Instruct --ffmpeg /path/to/ffmpeg
"""

import argparse
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ppe_compliance.config import DetectorConfig, VLMConfig
from ppe_compliance.detection.detector import PPEDetector
from ppe_compliance.ui.drawing import crop_box
from ppe_compliance.pipeline import process_video, video_fps
from ppe_compliance.tracking.temporal import ItemState, TemporalConfig
from ppe_compliance.reporting.vlm import VLMReporter

logger = logging.getLogger("make_demo_video")

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA
WHITE, LIGHT, DIM = (255, 255, 255), (215, 215, 215), (150, 150, 150)
RED, YELLOW, GREEN = (60, 60, 235), (0, 215, 255), (90, 220, 90)
BG = (28, 24, 22)


def torch_threads() -> int:
    """Return the number of CPU threads used by torch."""
    import torch

    return int(torch.get_num_threads())


def torch_device() -> str:
    """Return "GPU" when torch sees a CUDA device, else "CPU" (shown on the title card)."""
    import torch

    return "GPU" if torch.cuda.is_available() else "CPU"


def ascii_text(text: str) -> str:
    """Replace characters the Hershey fonts cannot draw.

    Args:
        text: Any text.

    Returns:
        ASCII-only text.
    """
    for src, dst in (("—", "-"), ("–", "-"), ("…", "..."), ("×", "x"),
                     ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def wrap(text: str, max_width: int, scale: float, thickness: int) -> List[str]:
    """Split text into lines that fit ``max_width`` pixels.

    Args:
        text: Text to wrap.
        max_width: Maximum line width in pixels.
        scale: Font scale.
        thickness: Font thickness.

    Returns:
        Wrapped lines.
    """
    lines: List[str] = []
    for paragraph in ascii_text(text).split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if cv2.getTextSize(candidate, FONT, scale, thickness)[0][0] <= max_width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def put_lines(img: np.ndarray, lines: Sequence[str], x: int, y: int, scale: float,
              color: Tuple[int, int, int], thickness: int = 2, gap: float = 1.55,
              center_width: Optional[int] = None) -> int:
    """Draw lines of text (anti-aliased) and return the y below the last line.

    Args:
        img: Image to draw on.
        lines: Text lines.
        x: Left x (or left edge of the centering area).
        y: Baseline of the first line.
        scale: Font scale.
        color: BGR color.
        thickness: Font thickness.
        gap: Line spacing as a multiple of the text height.
        center_width: If given, center each line in ``[x, x + center_width]``.

    Returns:
        The baseline y for the next line.
    """
    height = cv2.getTextSize("Ag", FONT, scale, thickness)[0][1]
    for line in lines:
        lx = x
        if center_width is not None:
            lx = x + (center_width - cv2.getTextSize(line, FONT, scale, thickness)[0][0]) // 2
        cv2.putText(img, line, (lx, y), FONT, scale, color, thickness, AA)
        y += int(height * gap)
    return y


def blend_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.82) -> None:
    """Draw a semi-transparent dark rectangle in place."""
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(BG) * alpha).astype(np.uint8)


def card(size: Tuple[int, int], title: str, lines: Sequence[Tuple[str, float, Tuple[int, int, int]]]) -> np.ndarray:
    """Render a title / end card.

    Args:
        size: ``(width, height)``.
        title: Big title text.
        lines: ``(text, scale, color)`` lines drawn under the title.

    Returns:
        The BGR card image.
    """
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    s = h / 1080
    title_lines = wrap(title, int(w * 0.8), 1.6 * s, 3)
    block = len(title_lines) * 1.6 * s * 55 + sum(sc * s * 60 for _, sc, _ in lines) + 60 * s
    y = int((h - block) / 2 + 40 * s)
    y = put_lines(img, title_lines, int(w * 0.1), y, 1.6 * s, WHITE, 3, 1.6, int(w * 0.8))
    y += int(30 * s)
    cv2.line(img, (int(w * 0.3), y - int(25 * s)), (int(w * 0.7), y - int(25 * s)), YELLOW, 2, AA)
    for text, scale, color in lines:
        y = put_lines(img, wrap(text, int(w * 0.8), scale * s, 2), int(w * 0.1), y + int(10 * s),
                      scale * s, color, 2, 1.7, int(w * 0.8))
    return img


def playback_overlay(frame: np.ndarray, t: float) -> np.ndarray:
    """Add a small legend bar to a normal playback frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    blend_rect(out, 0, h - 54, w, h, 0.7)
    x = 24
    for text, color in (("gray = checking (3 s window)", LIGHT), ("green = OK", GREEN),
                        ("red = confirmed violation", RED)):
        cv2.putText(out, text, (x, h - 19), FONT, 0.75, color, 2, AA)
        x += cv2.getTextSize(text, FONT, 0.75, 2)[0][0] + 48
    label = f"t = {t:5.1f} s"
    cv2.putText(out, label, (w - 24 - cv2.getTextSize(label, FONT, 0.75, 2)[0][0], h - 19),
                FONT, 0.75, LIGHT, 2, AA)
    return out


def freeze_frame(base: np.ndarray, box: Sequence[int], crop: np.ndarray, check: Dict[str, Any],
                 status_lines: Sequence[Tuple[str, float, Tuple[int, int, int]]]) -> np.ndarray:
    """Render one frozen frame with the highlighted person and the VLM panel.

    Args:
        base: Annotated frame of the moment of the freeze.
        box: Person box ``(x1, y1, x2, y2)``.
        crop: Raw person crop sent to the VLM.
        check: Check metadata (``number``, ``total``, ``track_id``, ``violations``, ``model``).
        status_lines: ``(text, scale, color)`` lines for the status area.

    Returns:
        The rendered BGR frame.
    """
    h, w = base.shape[:2]
    out = (base * 0.45).astype(np.uint8)
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    out[y1:y2, x1:x2] = base[y1:y2, x1:x2]
    cv2.rectangle(out, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), YELLOW, 7, AA)

    pw, margin = int(w * 0.34), 30
    px = margin if (x1 + x2) / 2 > w / 2 else w - pw - margin
    py1, py2 = margin, h - margin
    blend_rect(out, px, py1, px + pw, py2)
    cv2.rectangle(out, (px, py1), (px + pw, py2), YELLOW, 2, AA)
    ix, inner = px + 28, pw - 56
    y = put_lines(out, ["VLM double-check"], ix, py1 + 60, 1.25, YELLOW, 3)
    y = put_lines(out, [f"Person #{check['track_id']}   |   check {check['number']} of {check['total']}"],
                  ix, y, 0.75, LIGHT, 2)

    ch, cw = crop.shape[:2]
    box_w, box_h = inner, int(h * 0.36)
    scale = min(box_w / cw, box_h / ch)
    rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_CUBIC)
    cx, cy = ix + (box_w - rw) // 2, y
    out[cy:cy + rh, cx:cx + rw] = resized
    cv2.rectangle(out, (cx - 2, cy - 2), (cx + rw + 1, cy + rh + 1), DIM, 1, AA)
    y = cy + box_h + 50

    y = put_lines(out, ["Confirmed by the temporal rule:"], ix, y, 0.75, LIGHT, 2)
    y = put_lines(out, wrap(", ".join(check["violations"]), inner, 0.95, 2), ix, y + 4, 0.95, RED, 2)
    y += 14
    for text, sc, color in status_lines:
        y = put_lines(out, wrap(text, inner, sc, 2), ix, y + 6, sc, color, 2)
    put_lines(out, wrap(f"{check['model']} - model output, requires human review", inner, 0.6, 1),
              ix, py2 - 48, 0.6, DIM, 1)
    return out


class RecordingDetector:
    """Wraps :class:`PPEDetector` and remembers the detections of the last frame."""

    def __init__(self, detector: PPEDetector) -> None:
        """Wrap a detector.

        Args:
            detector: Detector used by the pipeline.
        """
        self._detector = detector
        self.spec = detector.spec
        self.last_detections: List[Any] = []
        self.last_frame: Optional[np.ndarray] = None

    def process_frame(self, frame: np.ndarray, frame_number: int, targets: Any = None) -> Any:
        """Delegate to the detector and store the frame and detections."""
        annotated, detections, logs = self._detector.process_frame(frame, frame_number, targets)
        self.last_frame, self.last_detections = frame, detections
        return annotated, detections, logs


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", default="outputs/demo_input.mp4", help="Input video.")
    p.add_argument("--output", default="results/demo_vlm_doublecheck.mp4", help="Final H.264 video.")
    p.add_argument("--raw-output", default="outputs/demo_vlm_raw.mp4", help="Intermediate mp4v video.")
    p.add_argument("--log", default=None, help="JSON log of the VLM checks (default: --output with .json).")
    p.add_argument("--vlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct", help="Hugging Face VLM id.")
    p.add_argument("--max-vlm-checks", type=int, default=3, help="Maximum number of VLM double-checks.")
    p.add_argument("--freeze-fps", type=float, default=10.0,
                   help="How often per second of video the elapsed counter is refreshed while frozen.")
    p.add_argument("--max-wait-seconds", type=float, default=4.0,
                   help="Video time used to show the VLM wait (the counter shows the real time).")
    p.add_argument("--answer-seconds", type=float, default=3.0, help="Video time the VLM answer is shown.")
    p.add_argument("--min-crop-height", type=int, default=150,
                   help="Persons with a smaller box are not sent to the VLM.")
    p.add_argument("--min-gap-seconds", type=float, default=3.0,
                   help="Minimum input time between two checks.")
    p.add_argument("--title", default="PPE compliance check — YOLO detection + temporal confirmation "
                                      "+ VLM double-check", help="Title card text.")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable used for the final encode.")
    p.add_argument("--height", type=int, default=720, help="Output height of the final video.")
    p.add_argument("--crf", type=int, default=24, help="x264 CRF of the final video.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Render the demo video.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")
    fps = video_fps(cap)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    vlm_name = args.vlm_model.split("/")[-1]
    reporter = VLMReporter(VLMConfig(model_id=args.vlm_model))
    start = time.perf_counter()
    reporter.load()  # fail early, before the long detection loop
    load_seconds = time.perf_counter() - start
    logger.info("VLM loaded in %.1f s", load_seconds)

    detector = RecordingDetector(PPEDetector(DetectorConfig()))
    Path(args.raw_output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.raw_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer: {args.raw_output}")

    def hold(img: np.ndarray, seconds: float) -> None:
        for _ in range(max(1, round(seconds * fps))):
            writer.write(img)

    hold(card(size, args.title, [
        ("PPE: SafetyVision v2 (YOLOv8s)   |   Persons: YOLOv8n + ByteTrack + re-ID", 0.9, LIGHT),
        (f"Double-check: {vlm_name} ({torch_device()})", 0.9, LIGHT),
        ("Temporal rule: 3 s window, 60% majority. VLM waits shortened in the video.", 0.7, DIM),
    ]), 2.5)

    checks: List[Dict[str, Any]] = []
    seen: set = set()
    last_check_t = -math.inf
    for step in process_video(detector, args.input, None, TemporalConfig()):
        t = (step.index - 1) / fps
        writer.write(playback_overlay(step.annotated, t))
        if step.index % 100 == 0:
            logger.info("Frame %d/%d, %d check(s)", step.index, step.total, len(checks))
        for det, status in zip(detector.last_detections, step.persons):
            # status.track_id is the persistent person id (re-ID), so a person is checked once
            if not status.identified or status.state is not ItemState.VIOLATION or status.track_id in seen:
                continue
            if len(checks) >= args.max_vlm_checks or t - last_check_t < args.min_gap_seconds:
                continue
            if det.box[3] - det.box[1] < args.min_crop_height:
                continue
            crop = crop_box(detector.last_frame, det.box)
            if crop.size == 0:
                continue
            seen.add(status.track_id)
            last_check_t = t
            info = {"number": len(checks) + 1, "total": args.max_vlm_checks, "track_id": status.track_id,
                    "violations": list(status.confirmed_violations), "model": vlm_name}
            logger.info("VLM check %d: person #%d %s", info["number"], status.track_id, info["violations"])
            began = time.perf_counter()
            answer = reporter.describe(crop, status.confirmed_violations)
            elapsed = time.perf_counter() - began
            logger.info("VLM answered in %.1f s: %s", elapsed, answer)

            shown = min(elapsed, args.max_wait_seconds)
            n_wait = max(1, round(shown * fps))
            refresh = max(1, round(fps / args.freeze_fps))
            note = [("(wait shortened in the video)", 0.6, DIM)] if elapsed > shown + 0.05 else []
            for i in range(n_wait):
                k = min(n_wait, (i // refresh + 1) * refresh)
                value = elapsed * k / n_wait
                dots = "." * (1 + (i // max(1, round(fps / 3))) % 3)
                hold_img = freeze_frame(step.annotated, det.box, crop, info,
                                        [(f"Analyzing{dots:<3}  {value:5.1f} s", 1.0, WHITE)] + note)
                writer.write(hold_img)
            hold(freeze_frame(step.annotated, det.box, crop, info, [
                (f"VLM answer ({elapsed:.1f} s):", 0.75, LIGHT), (f'"{answer}"', 0.85, WHITE)]),
                args.answer_seconds)
            checks.append({"check": info["number"], "person_id": status.track_id, "track_id": det.track_id, "frame_index": step.index,
                           "input_time_s": round(t, 2), "violations": info["violations"],
                           "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                           "elapsed_seconds": round(elapsed, 2), "answer": answer})

    hold(card(size, "Models and credits", [
        ("PPE detection: SafetyVision v2 (YOLOv8s), AGPL-3.0", 0.85, LIGHT),
        ("Person detection + tracking: YOLOv8n + ByteTrack (Ultralytics), AGPL-3.0", 0.85, LIGHT),
        (f"VLM double-check: {vlm_name}", 0.85, LIGHT),
        ("Footage: Pexels (Pexels License)", 0.85, LIGHT),
        ("Detections and VLM text are model outputs and require human review.", 0.8, YELLOW),
        ("Prepared with AI assistance.", 0.7, DIM),
    ]), 3.0)
    writer.release()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", args.raw_output,
                    "-vf", f"scale=-2:{args.height}", "-c:v", "libx264", "-preset", "medium",
                    "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    args.output], check=True)
    log_path = Path(args.log or Path(args.output).with_suffix(".json"))
    log_path.write_text(json.dumps({
        "input": str(args.input), "output": str(args.output), "vlm_model": args.vlm_model,
        "vlm_load_seconds": round(load_seconds, 2), "torch_threads": torch_threads(),
        "device": torch_device(),
        "ppe_model": DetectorConfig().ppe_model, "temporal": asdict(TemporalConfig()),
        "checks": checks,
        "note": "Detections and VLM answers are model outputs; human review required.",
    }, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", args.output, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
