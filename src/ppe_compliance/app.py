"""Streamlit UI: upload a video, watch the annotated feed, read the VLM reports.

Launch with ``ppe-compliance`` or ``streamlit run src/ppe_compliance/app.py -- [options]``.
"""

import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):  # executed by `streamlit run` as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from ppe_compliance.aggregation import PersonInfractionAggregator, total_infractions
from ppe_compliance.cli import build_app_parser, build_configs
from ppe_compliance.config import available_trackers
from ppe_compliance.detector import PPEDetector
from ppe_compliance.drawing import bgr_to_pil, bgr_to_rgb
from ppe_compliance.models import PPE_MODELS
from ppe_compliance.pipeline import process_video
from ppe_compliance.vlm import VLMReporter, fallback_description

MAX_LOG_LINES = 20
UI_REFRESH_S = 0.1

args = build_app_parser().parse_args(sys.argv[1:])
detector_cfg, report_cfg, vlm_cfg, temporal_cfg = build_configs(args)


@st.cache_resource
def load_detector(ppe_model: str) -> PPEDetector:
    """Load the detection models once per Streamlit server and PPE model."""
    return PPEDetector(replace(detector_cfg, ppe_model=ppe_model))


@st.cache_resource
def load_reporter(model_id: str) -> VLMReporter:
    """Create the (lazily loaded) VLM reporter once per Streamlit server."""
    return VLMReporter(replace(vlm_cfg, model_id=model_id))


st.set_page_config(page_title="PPE Compliance VLM", layout="wide", initial_sidebar_state="expanded")
st.title("PPE Compliance Monitoring with VLM Reports")

st.sidebar.header("Inspection settings")
model_keys = list(PPE_MODELS)
ppe_model = st.sidebar.selectbox(
    "PPE model:",
    model_keys,
    index=model_keys.index(args.ppe_model),
    format_func=lambda k: PPE_MODELS[k].display_name,
)
spec = PPE_MODELS[ppe_model]
st.sidebar.caption(f"Source: {spec.source_url} ({spec.license}), imgsz {spec.imgsz}")
supported = spec.infraction_labels()
selected = st.sidebar.multiselect(
    "Infractions to monitor:",
    options=list(supported),
    default=list(supported),
    format_func=lambda x: supported[x],
    key=f"infractions_{ppe_model}",  # separate selection per model
)
detector = load_detector(ppe_model)
trackers = available_trackers()
if args.tracker not in trackers:
    trackers.insert(0, args.tracker)
tracker = st.sidebar.selectbox("Tracker configuration:", trackers, index=trackers.index(args.tracker))
st.sidebar.caption(
    f"A person turns red or green only after a {temporal_cfg.window_seconds:g} s majority "
    "of frames agrees (gray while checking)."
)
st.sidebar.caption("Detections and VLM text are model outputs and require human review.")

col_stream, col_reports = st.columns([3, 2])
with col_stream:
    st.subheader("Live processing feed")
    uploaded = st.file_uploader("Upload a video for analysis", type=["mp4", "avi", "mov"])
    status_text = st.empty()
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Tracking and association logs")
    log_box = st.empty()
with col_reports:
    st.subheader("Infraction reports (generated at the end)")
    reports_area = st.container()

state = st.session_state
state.setdefault("aggregator", None)
state.setdefault("cache_key", None)
state.setdefault("vlm_results", {})

if uploaded is None:
    status_text.info("Waiting for a video upload to start the inspection.")
    state.aggregator, state.cache_key, state.vlm_results = None, None, {}
    st.stop()

# The video is processed once per (file, model, tracker) with ALL infractions the
# model supports. Changing the infraction filter only re-selects the reports from
# the cached aggregator, so Streamlit reruns never process the same video twice.
key = (uploaded.name, uploaded.size, ppe_model, tracker)
if state.cache_key != key:  # new video, model or tracker: invalidate cached results
    state.aggregator, state.cache_key, state.vlm_results = None, key, {}

if state.aggregator is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
        tmp.write(uploaded.getvalue())
    detector.reset_tracking(tracker)
    aggregator = PersonInfractionAggregator()
    recent_logs: list = []
    last_ui = 0.0
    status_text.info("Processing video...")
    try:  # a Streamlit rerun interrupts this loop: always remove the temporary file
        for step in process_video(detector, tmp.name, None, temporal_cfg):
            aggregator.update_many(step.infractions)
            recent_logs = (recent_logs + step.logs)[-MAX_LOG_LINES:]
            now = time.time()
            if now - last_ui > UI_REFRESH_S or step.index == step.total:
                frame_placeholder.image(bgr_to_rgb(step.annotated), channels="RGB", width="stretch")
                progress_bar.progress(min(step.index / step.total, 1.0))
                log_box.text_area("Live logs", "\n".join(recent_logs), height=250, label_visibility="collapsed")
                last_ui = now
    except IOError as exc:
        status_text.error(str(exc))
        st.stop()
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    progress_bar.progress(1.0)
    state.aggregator = aggregator

status_text.success("Video processed (cached). Changing the infraction filter does not reprocess it.")
reports = state.aggregator.select_reports(report_cfg.min_appearances, report_cfg.min_ratio, labels=selected)
with reports_area:
    st.markdown("---")
    st.metric("Persons with infractions", len(reports))
    st.metric("Total consolidated infractions", total_infractions(reports))
    st.markdown("---")
    if not reports:
        st.info("No significant infractions detected for the selected filters.")
        st.stop()

    reporter = None if args.no_vlm else load_reporter(vlm_cfg.model_id)
    with st.status("Analyzing infractions with the VLM...", expanded=True) as status_box:
        for idx, report in enumerate(reports, start=1):
            vlm_key = (report.person_id, tuple(report.infractions))
            if vlm_key in state.vlm_results:
                continue
            status_box.update(label=f"Describing person #{report.person_id} ({idx}/{len(reports)})...")
            if reporter is None:
                text = fallback_description(report.infractions)
            else:
                try:
                    text = reporter.describe(report.crop, report.infractions)
                except Exception as exc:  # noqa: BLE001 - shown to the user
                    st.error(f"{exc} Using the template sentence instead.")
                    reporter = None  # do not retry a failing model for every person
                    text = fallback_description(report.infractions)
            state.vlm_results[vlm_key] = text
        status_box.update(label=f"Reports ready for {len(reports)} person(s).", state="complete", expanded=False)

    for report in reports:
        exp = st.expander(
            f"Report - person #{report.person_id} ({len(report.infractions)} infraction(s))", expanded=True
        )
        c1, c2 = exp.columns([1, 2])
        c1.image(bgr_to_pil(report.crop), caption=f"ID #{report.person_id}", width=130)
        c2.markdown("**Missing PPE / events:**\n" + "\n".join(f"- **{p}**" for p in report.infractions))
        vlm_text = state.vlm_results.get((report.person_id, tuple(report.infractions)), "Analysis pending...")
        c2.info(f"**VLM report:** {vlm_text}")
