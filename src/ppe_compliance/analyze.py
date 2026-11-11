"""Headless analysis: process a video and write the consolidated report as JSON.

Usage::

    python -m ppe_compliance analyze --input video.mp4 --output report.json [--no-vlm]
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregation import PersonInfractionAggregator, total_infractions
from .cli import add_pipeline_args, build_configs, build_reid_config
from .detector import PPEDetector
from .models import PPE_MODELS, get_model_spec, parse_infractions
from .pipeline import process_video, video_fps
from .vlm import VLMReporter, fallback_description

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``analyze`` argument parser.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ppe-compliance analyze",
        description="Process a video without UI and write a JSON report of the "
                    "temporally confirmed infractions (no crops).",
    )
    parser.add_argument("--input", required=True, help="Input video file.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--infractions", nargs="+", default=None, metavar="LABEL",
                        help="Infractions to monitor (default: all supported by --ppe-model; "
                             + "; ".join(f"{k}: {', '.join(s.infraction_labels())}"
                                         for k, s in PPE_MODELS.items())
                             + ").")
    parser.add_argument("--save-video", default=None, metavar="PATH",
                        help="Also write the annotated video (mp4v) to PATH.")
    add_pipeline_args(parser)
    return parser


def _open_writer(path: str, input_path: str, first_frame: Any) -> Any:
    """Create a ``cv2.VideoWriter`` matching the input FPS and the frame size.

    Args:
        path: Output video path (parent folders are created).
        input_path: Input video, used to read its frame rate (see
            :func:`pipeline.video_fps`).
        first_frame: First annotated frame, used for the output size.

    Returns:
        An opened ``cv2.VideoWriter``.

    Raises:
        IOError: If the writer cannot be opened.
    """
    import os

    import cv2

    cap = cv2.VideoCapture(input_path)
    fps = video_fps(cap)
    cap.release()
    height, width = first_frame.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer: {path}")
    return writer


def main(argv: Optional[List[str]] = None) -> int:
    """Run the headless analysis.

    Args:
        argv: Arguments without the program/subcommand name.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        targets = parse_infractions(args.infractions, get_model_spec(args.ppe_model))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        detector_cfg, report_cfg, vlm_cfg, temporal_cfg = build_configs(args)
        reid_cfg = build_reid_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not Path(args.input).is_file():
        parser.error(f"input video not found: {args.input}")

    detector = PPEDetector(detector_cfg)
    aggregator = PersonInfractionAggregator()
    frames = 0
    tracks_seen: set = set()
    reidentified = 0
    writer = None
    try:
        for step in process_video(detector, args.input, targets, temporal_cfg, reid_cfg):
            aggregator.update_many(step.infractions)
            reidentified += sum(d.reused for d in step.reid_decisions)
            tracks_seen.update(d.track_id for d in step.reid_decisions)
            frames = step.index
            if args.save_video:
                if writer is None:
                    writer = _open_writer(args.save_video, args.input, step.annotated)
                writer.write(step.annotated)
            if step.index % 100 == 0:
                logger.info("Processed %d/%d frames", step.index, step.total)
    finally:
        if writer is not None:
            writer.release()
            logger.info("Annotated video saved to %s", args.save_video)
    if args.save_video and writer is None:
        logger.warning("No frame decoded: annotated video not written (%s)", args.save_video)

    reports = aggregator.select_reports(report_cfg.min_appearances, report_cfg.min_ratio)
    reporter = None if args.no_vlm else VLMReporter(vlm_cfg)
    persons: List[Dict[str, Any]] = []
    for report in reports:
        if reporter is None:
            description = fallback_description(report.infractions)
        else:
            description = reporter.describe(report.crop, report.infractions)
        persons.append({
            "person_id": report.person_id,
            "track_ids": report.track_ids,
            "appearances": report.appearances,
            "infractions": report.infractions,
            "infraction_counts": report.ppe_counts,
            "description": description,
            "description_source": "template" if reporter is None else vlm_cfg.model_id,
        })

    result = {
        "input": str(args.input),
        "frames_processed": frames,
        "monitored_infractions": targets,
        "settings": {
            "ppe_model": detector_cfg.ppe_model,
            "tracker": detector_cfg.tracker,
            "min_appearances": report_cfg.min_appearances,
            "min_ratio": report_cfg.min_ratio,
            "ppe_conf": detector_cfg.ppe_conf,
            "temporal": asdict(temporal_cfg),
            "reid": asdict(reid_cfg),
        },
        "reid_summary": {
            "tracks_assigned": len(tracks_seen),
            "tracks_reidentified": reidentified,
            "persons": len(tracks_seen) - reidentified,
        } if reid_cfg.enabled else None,
        "persons_with_infractions": len(reports),
        "total_consolidated_infractions": total_infractions(reports),
        "persons": persons,
        "note": "Model outputs; human review required before any decision.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Report written to %s (%d person(s))", out, len(reports))
    return 0
