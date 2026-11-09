"""Headless analysis: process a video and write the consolidated report as JSON.

Usage::

    python -m ppe_compliance analyze --input video.mp4 --output report.json [--no-vlm]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregation import PersonInfractionAggregator, total_infractions
from .cli import add_pipeline_args, build_configs
from .detector import PPEDetector
from .models import PPE_MODELS, get_model_spec, parse_infractions
from .pipeline import process_video
from .vlm import VLMReporter, fallback_description

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``analyze`` argument parser.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ppe-compliance analyze",
        description="Process a video without UI and write a JSON infraction report (no crops).",
    )
    parser.add_argument("--input", required=True, help="Input video file.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--infractions", nargs="+", default=None, metavar="LABEL",
                        help="Infractions to monitor (default: all supported by --ppe-model; "
                             + "; ".join(f"{k}: {', '.join(s.infraction_labels())}"
                                         for k, s in PPE_MODELS.items())
                             + ").")
    add_pipeline_args(parser)
    return parser


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
    detector_cfg, report_cfg, vlm_cfg = build_configs(args)

    detector = PPEDetector(detector_cfg)
    aggregator = PersonInfractionAggregator()
    frames = 0
    for step in process_video(detector, args.input, targets):
        aggregator.update_many(step.infractions)
        frames = step.index
        if step.index % 100 == 0:
            logger.info("Processed %d/%d frames", step.index, step.total)

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
        },
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
