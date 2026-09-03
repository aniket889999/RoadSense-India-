import streamlit as st
import pandas as pd
import csv
from src.config import load_config
from src.video_io import validate_video, process_and_create_kit
from src.analysis_runner import run_analysis
from src.ml.experimental_inference import (
    ExperimentalInferenceSettings,
)
from src.ml.drive_review import build_drive_review_plan, run_verified_drive_review
from src.ml.local_model_runtime import (
    LocalModelRuntimeError,
    load_verified_yolo_model,
    validate_requested_device,
)
from src.ml.model_provenance import (
    FrozenBaselineVerificationError,
    load_frozen_baseline_config,
    verify_frozen_baseline,
)
import zipfile
from io import BytesIO
import hashlib
import math
from pathlib import Path

st.set_page_config(
    page_title="RoadSense India · Dashcam Review",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

config = load_config()
REPO_ROOT = Path(__file__).resolve().parent


st.markdown(
    """
    <style>
      :root {
        --rs-ink: #13251f;
        --rs-muted: #66766f;
        --rs-green: #0e7a50;
        --rs-lime: #b7ed62;
        --rs-mist: #eef5f0;
        --rs-line: #dce8e0;
      }
      .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(183, 237, 98, .14), transparent 28rem),
          linear-gradient(180deg, #f8fbf9 0%, #f2f7f4 100%);
        color: var(--rs-ink);
      }
      [data-testid="stHeader"] { background: rgba(248, 251, 249, .86); }
      [data-testid="stMainBlockContainer"] { max-width: 1380px; padding-top: 2rem; }
      .rs-hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.25rem;
        padding: clamp(1.4rem, 3.5vw, 3.2rem);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 28px;
        background: linear-gradient(128deg, #102d24 0%, #174f3b 64%, #227558 100%);
        box-shadow: 0 22px 65px rgba(18, 61, 46, .17);
        color: white;
      }
      .rs-hero::after {
        position: absolute;
        right: -5rem;
        bottom: -8rem;
        width: 25rem;
        height: 25rem;
        border: 2.5rem solid rgba(183, 237, 98, .12);
        border-radius: 50%;
        content: "";
      }
      .rs-kicker {
        margin: 0 0 .65rem;
        color: var(--rs-lime);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
      }
      .rs-hero h1 {
        position: relative;
        z-index: 1;
        max-width: 830px;
        margin: 0;
        color: white;
        font-size: clamp(2.4rem, 5vw, 5rem);
        line-height: .98;
        letter-spacing: -.065em;
      }
      .rs-hero h1 span { color: var(--rs-lime); }
      .rs-hero-copy {
        position: relative;
        z-index: 1;
        max-width: 720px;
        margin: 1rem 0 1.35rem;
        color: #d7e8df;
        font-size: clamp(.96rem, 1.5vw, 1.14rem);
        line-height: 1.55;
      }
      .rs-chip-row { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: .55rem; }
      .rs-chip {
        padding: .45rem .75rem;
        border: 1px solid rgba(218, 244, 228, .18);
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        color: #e9f5ee;
        font-size: .76rem;
        font-weight: 650;
      }
      .rs-chip-live { background: rgba(183,237,98,.14); color: #d7ff91; }
      .rs-section-title { margin: 1.8rem 0 .25rem; color: var(--rs-ink); font-size: 1.45rem; font-weight: 760; }
      .rs-section-copy { max-width: 850px; margin: 0 0 1rem; color: var(--rs-muted); line-height: 1.55; }
      .rs-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin: 1rem 0 1.3rem;
      }
      .rs-step {
        padding: 1rem 1.05rem;
        border: 1px solid var(--rs-line);
        border-radius: 16px;
        background: rgba(255,255,255,.78);
        box-shadow: 0 8px 24px rgba(37, 75, 59, .05);
      }
      .rs-step b { display: block; margin-bottom: .28rem; color: var(--rs-green); font-size: .8rem; }
      .rs-step span { color: #52635b; font-size: .84rem; line-height: 1.4; }
      .rs-trust {
        margin: .7rem 0 1.2rem;
        padding: .9rem 1rem;
        border-left: 4px solid var(--rs-green);
        border-radius: 10px;
        background: #edf8f1;
        color: #315c48;
        font-size: .86rem;
        line-height: 1.45;
      }
      .rs-trust strong { color: #174d37; }
      [data-testid="stFileUploader"] {
        padding: .55rem;
        border: 1px solid var(--rs-line);
        border-radius: 18px;
        background: rgba(255,255,255,.72);
      }
      [data-testid="stMetric"] {
        padding: .8rem 1rem;
        border: 1px solid var(--rs-line);
        border-radius: 16px;
        background: rgba(255,255,255,.82);
      }
      .stButton > button, .stDownloadButton > button {
        min-height: 2.65rem;
        border-radius: 12px;
        font-weight: 700;
      }
      .stButton > button[kind="primary"] {
        border-color: #0e7a50;
        background: #0e7a50;
      }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .35rem; }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1rem;
        border-radius: 12px 12px 0 0;
        font-weight: 700;
      }
      .rs-zero-note {
        padding: 1rem;
        border: 1px solid #f1d8a4;
        border-radius: 14px;
        background: #fff9eb;
        color: #70551d;
      }
      @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
        .rs-hero { border-radius: 20px; }
        .rs-steps { grid-template-columns: 1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_annotation_kit_manifest(kit_zip: bytes):
    """Read the kit's authoritative successful-frame list for previews/inference."""

    expected_fields = ["frame_index", "timestamp_seconds", "frame_file", "width", "height"]
    try:
        with zipfile.ZipFile(BytesIO(kit_zip)) as zf:
            manifest_text = zf.read("frame_manifest.csv").decode("utf-8")
            reader = csv.DictReader(manifest_text.splitlines())
            if reader.fieldnames != expected_fields:
                raise ValueError("The annotation kit frame manifest has unexpected columns.")

            names = set(zf.namelist())
            records = []
            previous_index = -1
            for row in reader:
                frame_index = int(row["frame_index"])
                timestamp_seconds = float(row["timestamp_seconds"])
                frame_file = row["frame_file"]
                if frame_index < 0 or frame_index <= previous_index:
                    raise ValueError("The annotation kit frame manifest has invalid frame ordering.")
                if not frame_file.startswith("frame_") or "/" in frame_file or "\\" in frame_file:
                    raise ValueError("The annotation kit frame manifest contains an unsafe frame filename.")
                if f"frames/{frame_file}" not in names:
                    raise ValueError("A frame listed in the annotation kit manifest is missing from the kit.")
                previous_index = frame_index
                records.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp_seconds,
                        "frame_file": frame_file,
                    }
                )
    except (KeyError, UnicodeDecodeError, csv.Error, zipfile.BadZipFile) as exc:
        raise ValueError(f"Unable to read the annotation kit frame manifest: {exc}") from exc

    if not records:
        raise ValueError("The annotation kit has no successfully extracted sampled frames.")
    return records


def _load_session_verified_model(checkpoint_path: str, checkpoint_sha256: str):
    """Return a provenance-pinned model cached only within this Streamlit session.

    ``st.cache_resource`` is deliberately not used here: it is process-global,
    while Ultralytics model objects may retain mutable predictor state.  Keeping
    the object in ``st.session_state`` prevents different local browser sessions
    from sharing an inference object or any user-frame state.
    """

    identity = (checkpoint_path, checkpoint_sha256)
    cached = st.session_state.get("experimental_verified_model")
    if isinstance(cached, dict) and cached.get("identity") == identity:
        return cached["model"]

    model = load_verified_yolo_model(Path(checkpoint_path), checkpoint_sha256)
    st.session_state.experimental_verified_model = {"identity": identity, "model": model}
    return model


def _render_drive_review_panel(video_bytes, video_name, video_hash, metadata):
    """Render a bounded, upload-backed dashcam replay using the pinned model."""

    st.markdown('<p class="rs-section-title">Dashcam Drive Review</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rs-section-copy">Choose a short part of the recording and let the frozen local model scan '
        'sampled frames. Green circles make suggestions easy to see while you replay the result.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rs-trust"><strong>Recorded-video review:</strong> this is not a continuous live camera feed, '
        'driver warning, confirmed pothole report, or safety system. Green means “model suggestion”, not “verified”. '
        'The video and model stay on this computer.</div>',
        unsafe_allow_html=True,
    )

    try:
        frozen_config = load_frozen_baseline_config(
            config.experimental_inference.frozen_baseline_config_path,
            repo_root=REPO_ROOT,
        )
        model_info = verify_frozen_baseline(frozen_config, repo_root=REPO_ROOT)
    except (FrozenBaselineVerificationError, ValueError, OSError) as exc:
        st.error(f"Experimental local model is unavailable: {exc}")
        st.info("The manual annotation baseline remains available and does not require the model runtime.")
        return

    max_duration = max(
        1,
        min(
            int(config.drive_review.max_window_seconds),
            int(math.ceil(metadata.duration_sec)),
        ),
    )
    default_duration = min(int(config.drive_review.default_window_seconds), max_duration)

    control_a, control_b, control_c = st.columns([1, 1, 1.1])
    with control_a:
        if max_duration == 1:
            window_duration = 1
            st.metric("Review window", "1 second")
        else:
            window_duration = st.slider(
                "Review window (seconds)",
                min_value=1,
                max_value=max_duration,
                value=default_duration,
                step=1,
                key=f"drive_window_duration_{video_hash}",
            )
    with control_b:
        max_start = max(0.0, float(metadata.duration_sec) - float(window_duration))
        rounded_max_start = float(round(max_start, 2))
        if rounded_max_start >= 0.01:
            window_start = st.slider(
                "Start at (seconds)",
                min_value=0.0,
                max_value=rounded_max_start,
                value=0.0,
                step=max(0.01, min(1.0, rounded_max_start)),
                key=f"drive_window_start_{video_hash}_{window_duration}",
            )
        else:
            window_start = 0.0
            st.metric("Start at", "0.0 seconds")
    with control_c:
        selected_threshold = st.slider(
            "Suggestion threshold",
            min_value=0.05,
            max_value=0.80,
            value=min(
                0.80,
                max(0.05, float(config.experimental_inference.default_confidence_threshold)),
            ),
            step=0.05,
            key=f"drive_threshold_{video_hash}",
            help=(
                "A lower display threshold shows more possible potholes but also more false positives. "
                "It is not a calibrated safety score."
            ),
        )

    try:
        plan = build_drive_review_plan(
            frame_count=metadata.frame_count,
            source_fps=metadata.fps,
            window_start_seconds=float(window_start),
            window_duration_seconds=float(window_duration),
            sampling_fps=config.drive_review.sampling_fps,
            max_frames=config.drive_review.max_sampled_frames,
        )
    except ValueError as exc:
        st.error(f"Unable to create the selected Drive Review window: {exc}")
        return

    status_a, status_b, status_c = st.columns(3)
    status_a.metric("Local model", "Verified")
    status_b.metric("Frames to inspect", plan.sampled_frame_count)
    status_c.metric("Sampling cadence", f"{plan.requested_sampling_fps:g} FPS")
    st.caption(
        f"Pinned run `{model_info.run_id}` · checkpoint `{model_info.checkpoint_sha256[:12]}…` · "
        f"source window {plan.window_start_seconds:.1f}s–{plan.window_end_seconds:.1f}s"
    )

    result_key = (
        video_hash,
        model_info.checkpoint_sha256,
        model_info.model_metadata_sha256,
        config.experimental_inference.device,
        config.experimental_inference.image_size,
        float(selected_threshold),
        config.experimental_inference.iou_threshold,
        config.experimental_inference.max_detections_per_frame,
        config.drive_review.output_fps,
        plan.frame_indices,
    )
    stored_key = st.session_state.get("drive_review_result_key")
    stored_result = st.session_state.get("drive_review_result")

    if st.button(
        "Analyze selected drive window",
        type="primary",
        key=f"run_drive_review_{video_hash}_{window_start}_{window_duration}",
    ):
        try:
            device = validate_requested_device(config.experimental_inference.device)
            settings = ExperimentalInferenceSettings(
                device=device,
                image_size=config.experimental_inference.image_size,
                confidence_threshold=float(selected_threshold),
                iou_threshold=config.experimental_inference.iou_threshold,
                max_detections_per_frame=config.experimental_inference.max_detections_per_frame,
                output_fps=config.drive_review.output_fps,
            )
            with st.spinner(
                f"Scanning {plan.sampled_frame_count} frames locally on {device.upper()}…"
            ):
                model = _load_session_verified_model(
                    str(model_info.checkpoint_path), model_info.checkpoint_sha256
                )
                result = run_verified_drive_review(
                    video_bytes=video_bytes,
                    video_filename=video_name,
                    plan=plan,
                    model=model,
                    model_info=model_info,
                    settings=settings,
                    input_video_sha256=video_hash,
                )
            st.session_state.drive_review_result_key = result_key
            st.session_state.drive_review_result = result
            stored_key = result_key
            stored_result = result
        except (LocalModelRuntimeError, RuntimeError, ValueError, OSError) as exc:
            st.error(f"Drive Review did not run: {exc}")
            st.info("The original video, manual annotations, and reports were not changed.")

    if stored_key != result_key or stored_result is None:
        st.info("Select a short window, then run the local review. Nothing is uploaded to an external service.")
        return

    detections = list(stored_result.detections)
    result_a, result_b, result_c, result_d = st.columns(4)
    result_a.metric("Suggestions", len(detections))
    result_b.metric("Frames flagged", stored_result.frames_with_detections)
    result_c.metric("Frames reviewed", stored_result.total_sampled_frames)
    result_d.metric("Threshold", f"{selected_threshold:.2f}")

    if detections:
        st.info(
            "Suggestions were found. Inspect each green circle in the replay; only a person can confirm whether it is a pothole."
        )
    else:
        st.markdown(
            '<div class="rs-zero-note"><strong>No suggestion crossed this threshold.</strong> This does not prove the '
            'road has no potholes. Try a window where the pothole is larger/closer, or lower the display threshold '
            'toward 0.10 and expect more false positives.</div>',
            unsafe_allow_html=True,
        )

    try:
        with zipfile.ZipFile(BytesIO(stored_result.report_zip)) as zf:
            annotated_video = zf.read("annotated_experimental_predictions.mp4")
    except (KeyError, zipfile.BadZipFile):
        st.error("The Drive Review playback video is unavailable.")
        annotated_video = None

    if annotated_video:
        st.markdown("#### Annotated replay")
        st.video(annotated_video)
        st.caption(
            "Playback is made from sampled source frames, so its timing is not real-time. "
            "Green circles appear only around raw model suggestions."
        )

    if detections:
        suggestion_rows = [
            {
                "Time (s)": round(detection.timestamp_seconds, 2),
                "Frame": detection.frame_index,
                "Score": round(detection.confidence, 3),
                "Left": round(detection.x_min, 1),
                "Top": round(detection.y_min, 1),
                "Right": round(detection.x_max, 1),
                "Bottom": round(detection.y_max, 1),
            }
            for detection in detections
        ]
        st.markdown("#### Review queue")
        st.dataframe(pd.DataFrame(suggestion_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Drive Review ZIP",
        data=stored_result.report_zip,
        file_name="roadsense_drive_review.zip",
        mime="application/zip",
        key=f"download_drive_review_{video_hash}",
    )
    st.caption(
        "The ZIP contains the annotated sampled replay, raw per-frame suggestions, and path-free model provenance. "
        "It does not create confirmed incidents automatically."
    )

def _render_video_facts(metadata):
    fact_a, fact_b, fact_c, fact_d = st.columns(4)
    fact_a.metric("Duration", f"{metadata.duration_sec:.1f}s")
    fact_b.metric("Resolution", f"{metadata.width} × {metadata.height}")
    fact_c.metric("Source FPS", f"{metadata.fps:.1f}")
    fact_d.metric("Frames", f"{metadata.frame_count:,}")
    st.caption(
        f"{metadata.filename} · {metadata.size_mb:.2f} MB · "
        f"{metadata.sampled_frame_count} frames available in the manual annotation kit"
    )


def _render_manual_workspace(video_bytes, video_name, video_hash, metadata, kit_zip):
    st.markdown('<p class="rs-section-title">Human-verified evidence workflow</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rs-section-copy">Download sampled frames, draw the boxes yourself, then upload the strict CSV. '
        'Only these human-provided boxes can become incidents in the evidence report.</p>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Manual reports do not inherit model suggestions. RoadSense does not calculate pothole severity, repair priority, "
        "traffic volume, GPS location, or accident risk in this workflow."
    )

    st.download_button(
        "Download annotation kit",
        data=kit_zip,
        file_name="annotation_kit.zip",
        mime="application/zip",
        key=f"download_annotation_kit_{video_hash}",
    )

    with st.expander("Preview sampled manual-review frames"):
        try:
            manifest_records = _read_annotation_kit_manifest(kit_zip)
            with zipfile.ZipFile(BytesIO(kit_zip)) as zf:
                preview_records = manifest_records[:12]
                cols = st.columns(4)
                for idx, record in enumerate(preview_records):
                    img_bytes = zf.read(f"frames/{record['frame_file']}")
                    cols[idx % 4].image(
                        img_bytes,
                        caption=f"Frame {record['frame_index']} · {record['timestamp_seconds']:.2f}s",
                    )
                if len(manifest_records) > 12:
                    st.info(f"{len(manifest_records) - 12} additional frames are included in the kit.")
        except (ValueError, KeyError, zipfile.BadZipFile) as exc:
            st.error(f"Unable to preview sampled frames: {exc}")

    st.markdown("#### Upload completed annotations")
    uploaded_csv = st.file_uploader(
        "Upload your completed manual annotations CSV",
        type=["csv"],
        key=f"manual_csv_{video_hash}",
    )

    if not uploaded_csv:
        st.info("Complete the CSV from the annotation kit, then upload it here to create a human-verified report.")
        return

    csv_bytes = uploaded_csv.getvalue()
    with st.spinner("Validating manual annotations and generating evidence…"):
        try:
            incidents, errors, report_zip, summary = run_analysis(
                csv_bytes,
                video_bytes,
                video_name,
                metadata.width,
                metadata.height,
                metadata.fps,
                metadata.sampled_frame_indices,
            )
        except Exception as exc:
            st.error(f"An error occurred during manual analysis: {exc}")
            return

    if errors:
        st.error("The manual CSV needs correction:")
        st.dataframe(pd.DataFrame({"Validation error": errors}), hide_index=True, use_container_width=True)
        return

    st.success("Manual annotations validated. The report contains only reviewer-provided evidence.")
    result_a, result_b = st.columns(2)
    result_a.metric("Manual observations", summary.total_observations)
    result_b.metric("Reviewer-grouped incidents", summary.total_incidents)
    st.info(
        "Repeated observations are consolidated only by the incident IDs supplied by the reviewer; this is not ML tracking."
    )

    if not incidents:
        st.info("The valid CSV did not contain any incidents to display.")
        return

    st.dataframe(
        pd.DataFrame([incident.model_dump() for incident in incidents]),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Download human-verified report ZIP",
        data=report_zip,
        file_name="roadsense_manual_report.zip",
        mime="application/zip",
        key=f"download_manual_report_{video_hash}",
    )

    st.markdown("#### Evidence preview")
    try:
        with zipfile.ZipFile(BytesIO(report_zip)) as zf:
            first_incident = incidents[0]
            safe_name = "".join(
                character
                for character in first_incident.evidence_file
                if character.isalnum() or character in ("-", "_", ".")
            )
            try:
                st.image(
                    zf.read(f"evidence/{safe_name}"),
                    caption=f"Human-provided evidence · {first_incident.incident_id}",
                )
            except KeyError:
                st.error(f"Missing evidence image for {first_incident.incident_id}")
            try:
                st.video(zf.read("annotated_manual_samples.mp4"))
            except KeyError:
                st.error("Annotated manual-samples MP4 is missing from the report.")
    except zipfile.BadZipFile:
        st.error("Failed to read the generated report ZIP.")


def _initialise_session_state():
    defaults = {
        "last_uploaded_video_hash": None,
        "metadata": None,
        "kit_zip": None,
        "experimental_verified_model": None,
        "drive_review_result_key": None,
        "drive_review_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.markdown(
    """
    <section class="rs-hero">
      <p class="rs-kicker">Local dashcam intelligence · India road review</p>
      <h1>See the road.<br><span>Review what matters.</span></h1>
      <p class="rs-hero-copy">RoadSense India turns recorded dashcam footage into a short, visual pothole-review queue—then keeps a human in control of every confirmed report.</p>
      <div class="rs-chip-row">
        <span class="rs-chip rs-chip-live">● Local model ready</span>
        <span class="rs-chip">Private on-device processing</span>
        <span class="rs-chip">Green-circle replay</span>
        <span class="rs-chip">Human verification required</span>
      </div>
    </section>
    <div class="rs-steps">
      <div class="rs-step"><b>01 · Upload</b><span>Select a recorded road or dashcam video from this computer.</span></div>
      <div class="rs-step"><b>02 · Review</b><span>Scan a short window and replay raw suggestions with green circles.</span></div>
      <div class="rs-step"><b>03 · Confirm</b><span>Use the separate manual workflow for evidence-backed incidents.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

_initialise_session_state()

st.markdown('<p class="rs-section-title">Start with a recording</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="rs-section-copy">For this version, upload a recorded dashcam clip. Direct live-camera alerts are a later '
    'milestone because they require latency, device, and field-safety validation.</p>',
    unsafe_allow_html=True,
)
uploaded_video = st.file_uploader(
    "Upload a road video",
    type=["mp4", "mov", "avi"],
    help=f"Maximum upload size: {config.app.max_upload_mb} MB",
)

if uploaded_video is None:
    st.info("Choose a short daytime road video to begin. Your file is processed only by this local Streamlit session.")
else:
    video_bytes = uploaded_video.getvalue()
    video_hash = hashlib.sha256(video_bytes).hexdigest()

    if st.session_state.last_uploaded_video_hash != video_hash:
        st.session_state.last_uploaded_video_hash = video_hash
        st.session_state.metadata = None
        st.session_state.kit_zip = None
        st.session_state.drive_review_result_key = None
        st.session_state.drive_review_result = None

    is_valid, message = validate_video(uploaded_video.name, len(video_bytes), config)
    if not is_valid:
        st.error(message)
    else:
        with st.expander("Preview original recording", expanded=False):
            st.video(video_bytes)

        if st.session_state.kit_zip is None:
            st.info(
                "Prepare the recording once. RoadSense will read its metadata and create the bounded frame kit used by both workflows."
            )
            if st.button("Create annotation kit", type="primary"):
                with st.spinner("Reading video metadata and extracting bounded sample frames…"):
                    try:
                        metadata, kit_zip = process_and_create_kit(
                            video_bytes,
                            uploaded_video.name,
                            config,
                        )
                        st.session_state.metadata = metadata
                        st.session_state.kit_zip = kit_zip
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            metadata = st.session_state.metadata
            kit_zip = st.session_state.kit_zip
            _render_video_facts(metadata)

            drive_tab, manual_tab, about_tab = st.tabs(
                ["Drive Review", "Manual Evidence", "How RoadSense works"]
            )
            with drive_tab:
                if config.drive_review.enabled and config.experimental_inference.enabled:
                    _render_drive_review_panel(
                        video_bytes,
                        uploaded_video.name,
                        video_hash,
                        metadata,
                    )
                else:
                    st.info("Drive Review is disabled in the local configuration.")
            with manual_tab:
                _render_manual_workspace(
                    video_bytes,
                    uploaded_video.name,
                    video_hash,
                    metadata,
                    kit_zip,
                )
            with about_tab:
                st.markdown(
                    """
                    ### What the current app does

                    1. It reads a recorded road video locally.
                    2. The optional frozen YOLOv8n research baseline inspects selected sampled frames for one class: **pothole**.
                    3. Raw suggestions are drawn as green circles in a non-real-time replay.
                    4. A person reviews the suggestions. Only separately entered manual CSV boxes become human-verified incidents.

                    ### What it does not do

                    It does not continuously watch a live dashcam, warn a driver, measure pothole depth, estimate accident risk, locate a pothole by GPS, or automatically send a repair request. The frozen baseline has modest held-out performance and can miss visible potholes or circle non-pothole road texture.
                    """
                )

st.markdown(
    "<div style='height:2rem'></div><p style='text-align:center;color:#728078;font-size:.78rem'>RoadSense India · Research and portfolio baseline · Field verification required</p>",
    unsafe_allow_html=True,
)
