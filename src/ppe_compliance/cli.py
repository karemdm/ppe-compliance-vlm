"""Command-line entry point.

``ppe-compliance [options]`` launches the Streamlit app; options after the
command are forwarded to the app. ``ppe-compliance analyze ...`` runs the
headless analysis and writes a JSON report.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .config import CUSTOM_TRACKER, DetectorConfig, ReportConfig, VLMConfig
from .models import PPE_MODELS


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by the app and the headless analysis.

    Args:
        parser: Parser to extend.
    """
    d, r, v = DetectorConfig(), ReportConfig(), VLMConfig()
    parser.add_argument("--person-weights", default=d.person_weights,
                        help="Person detection weights (falls back to auto-download by file name).")
    parser.add_argument("--person-conf", type=float, default=d.person_conf,
                        help="Confidence threshold for person detections (raise it to drop false persons).")
    parser.add_argument("--ppe-conf", type=float, default=d.ppe_conf,
                        help="Confidence threshold for PPE / infraction detections.")
    parser.add_argument("--tracker", default=d.tracker,
                        help=f"Tracker name or YAML path, e.g. bytetrack.yaml or {CUSTOM_TRACKER}.")
    parser.add_argument("--ppe-model", choices=list(PPE_MODELS), default=d.ppe_model,
                        help="Bundled PPE model: "
                             + "; ".join(f"{k} = {s.display_name}" for k, s in PPE_MODELS.items())
                             + f" (default: {d.ppe_model}).")
    parser.add_argument("--min-appearances", type=int, default=r.min_appearances,
                        help="Minimum frames with an infraction for a person to be reported.")
    parser.add_argument("--min-ratio", type=float, default=r.min_ratio,
                        help="Minimum count/appearances ratio for an infraction to be reported.")
    parser.add_argument("--vlm-model", default=v.model_id, help="Hugging Face id of the VLM.")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip the VLM and use a template sentence in the reports.")


def build_configs(args: argparse.Namespace) -> Tuple[DetectorConfig, ReportConfig, VLMConfig]:
    """Create the configuration objects from parsed arguments.

    Args:
        args: Namespace produced by a parser extended with :func:`add_pipeline_args`.

    Returns:
        Detector, report and VLM configurations.
    """
    detector = DetectorConfig(
        person_weights=args.person_weights,
        person_conf=args.person_conf,
        ppe_conf=args.ppe_conf,
        tracker=args.tracker,
        ppe_model=args.ppe_model,
    )
    report = ReportConfig(min_appearances=args.min_appearances, min_ratio=args.min_ratio)
    return detector, report, VLMConfig(model_id=args.vlm_model)


def build_app_parser() -> argparse.ArgumentParser:
    """Build the parser of the Streamlit app options.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ppe-compliance",
        description="Launch the PPE compliance Streamlit app. "
                    "Use 'ppe-compliance analyze --help' for the headless mode.",
    )
    add_pipeline_args(parser)
    return parser


def launch_app(app_args: List[str]) -> int:
    """Run ``streamlit run app.py -- <app_args>`` in a subprocess.

    Args:
        app_args: Options forwarded to the app.

    Returns:
        The Streamlit process exit code.
    """
    app_path = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--", *app_args]
    logging.getLogger(__name__).info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch to the app launcher or to the ``analyze`` subcommand.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        Process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if argv and argv[0] == "analyze":
        from .analyze import main as analyze_main

        return analyze_main(argv[1:])
    if argv and argv[0] == "app":
        argv = argv[1:]
    build_app_parser().parse_args(argv)  # validate early, show --help
    return launch_app(argv)


if __name__ == "__main__":
    sys.exit(main())

