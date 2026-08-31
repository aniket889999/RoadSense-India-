import streamlit as st
import pandas as pd
import csv
from src.config import load_config
from src.video_io import validate_video, process_and_create_kit
from src.analysis_runner import run_analysis
from src.ml.experimental_inference import (
    ExperimentalInferenceSettings,
    run_verified_sampled_video_inference,
)
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
from pathlib import Path

st.set_page_config(page_title="RoadSense India", layout="wide")

config = load_config()
REPO_ROOT = Path(__file__).resolve().parent


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


def _render_experimental_inference_panel(video_bytes, video_name, video_hash, kit_zip):
    """Render a separate, opt-in UI that cannot enter the manual report pipeline."""

    st.markdown("---")
    st.subheader("Experimental Local Model Suggestions (Optional)")
    st.warning(
        "This optional feature runs the pinned local YOLOv8n research baseline only on the sampled frames. "
        "Its boxes are unverified suggestions—not confirmed potholes, incidents, safety advice, repair priorities, "
        "severity scores, traffic estimates, or field decisions. Review every box manually."
    )
    st.caption(
        "This feature runs locally. It does not train, download weights, call an external API, or re-evaluate the "
        "held-out test split. Its confidence threshold is only a review-display filter, not a calibrated risk score."
    )

    try:
        frozen_config_path = config.experimental_inference.frozen_baseline_config_path
        frozen_config = load_frozen_baseline_config(frozen_config_path, repo_root=REPO_ROOT)
        model_info = verify_frozen_baseline(frozen_config, repo_root=REPO_ROOT)
        manifest_records = _read_annotation_kit_manifest(kit_zip)
    except (FrozenBaselineVerificationError, ValueError, OSError) as exc:
        st.error(f"Experimental local model is unavailable: {exc}")
        st.info("The manual annotation baseline remains available and does not require the model runtime.")
        return

    sampled_indices = [record["frame_index"] for record in manifest_records]
    default_threshold = float(config.experimental_inference.default_confidence_threshold)
    selected_threshold = st.slider(
        "Review-display confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=min(0.95, max(0.05, default_threshold)),
        step=0.05,
        key=f"experimental_threshold_{video_hash}",
        help="This filter changes which raw model suggestions are shown. It was not tuned on the held-out test split.",
    )

    st.caption(
        f"Pinned local checkpoint verified: run `{model_info.run_id}` · SHA-256 `{model_info.checkpoint_sha256[:12]}…` "
        f"· {len(sampled_indices)} kit frames available."
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
        config.experimental_inference.output_fps,
        tuple(sampled_indices),
    )
    stored_key = st.session_state.get("experimental_inference_result_key")
    stored_result = st.session_state.get("experimental_inference_result")

    if st.button("Run experimental local suggestions", key=f"run_experimental_{video_hash}"):
        try:
            device = validate_requested_device(config.experimental_inference.device)
            settings = ExperimentalInferenceSettings(
                device=device,
                image_size=config.experimental_inference.image_size,
                confidence_threshold=float(selected_threshold),
                iou_threshold=config.experimental_inference.iou_threshold,
                max_detections_per_frame=config.experimental_inference.max_detections_per_frame,
                output_fps=config.experimental_inference.output_fps,
            )
            with st.spinner("Running local experimental suggestions on the sampled frames..."):
                model = _load_session_verified_model(
                    str(model_info.checkpoint_path), model_info.checkpoint_sha256
                )
                result = run_verified_sampled_video_inference(
                    video_bytes=video_bytes,
                    video_filename=video_name,
                    sampled_frame_indices=sampled_indices,
                    model=model,
                    model_info=model_info,
                    settings=settings,
                    input_video_sha256=video_hash,
                )
            st.session_state.experimental_inference_result_key = result_key
            st.session_state.experimental_inference_result = result
            stored_key = result_key
            stored_result = result
        except (LocalModelRuntimeError, RuntimeError, ValueError, OSError) as exc:
            st.error(f"Experimental local inference did not run: {exc}")
            st.info("No manual report or annotation data was changed.")

    if stored_key != result_key or stored_result is None:
        st.info("Run the optional model only when you want unverified suggestions for these sampled frames.")
        return

    st.success("Experimental local suggestions generated. They remain separate from manual annotations and reports.")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Unverified model suggestions", len(stored_result.detections))
    metric2.metric("Frames with suggestions", stored_result.frames_with_detections)
    metric3.metric("Sampled frames processed", stored_result.total_sampled_frames)

    if stored_result.detections:
        suggestion_rows = [
            {
                "frame_index": detection.frame_index,
                "original_time_seconds": round(detection.timestamp_seconds, 3),
                "model_score": round(detection.confidence, 4),
                "x_min": round(detection.x_min, 1),
                "y_min": round(detection.y_min, 1),
                "x_max": round(detection.x_max, 1),
                "y_max": round(detection.y_max, 1),
            }
            for detection in stored_result.detections
        ]
        st.dataframe(pd.DataFrame(suggestion_rows), use_container_width=True)
    else:
        st.info("No unverified model suggestions met the selected review-display threshold.")

    st.download_button(
        "Download Experimental Suggestions ZIP",
        data=stored_result.report_zip,
        file_name="roadsense_experimental_model_suggestions.zip",
        mime="application/zip",
    )
    try:
        with zipfile.ZipFile(BytesIO(stored_result.report_zip)) as zf:
            st.video(zf.read("annotated_experimental_predictions.mp4"))
    except (KeyError, zipfile.BadZipFile):
        st.error("The experimental prediction preview video is unavailable.")

    st.caption(
        "To create a RoadSense incident report, manually review accepted boxes and enter them with your own incident IDs "
        "in the human-provided CSV below. Model suggestions are never automatically converted into manual observations."
    )

st.title("RoadSense India")
st.subheader("Step 2: Manual Annotation Baseline")

st.markdown("""
RoadSense's primary workflow is a Manual Annotation Baseline.
It samples video frames and turns human-provided pothole boxes into evidence-backed incident reports.
""")

st.warning(
    "Manual incident reports use only human-provided CSV boxes. RoadSense does not calculate traffic volume or repair "
    "priority. An optional, clearly separate experimental local-model panel may appear after you create an annotation kit."
)

if "last_uploaded_video_hash" not in st.session_state:
    st.session_state.last_uploaded_video_hash = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "kit_zip" not in st.session_state:
    st.session_state.kit_zip = None
if "experimental_inference_result_key" not in st.session_state:
    st.session_state.experimental_inference_result_key = None
if "experimental_inference_result" not in st.session_state:
    st.session_state.experimental_inference_result = None
if "experimental_verified_model" not in st.session_state:
    st.session_state.experimental_verified_model = None

uploaded_video = st.file_uploader("Upload a road video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    video_bytes = uploaded_video.getvalue()
    video_hash = hashlib.sha256(video_bytes).hexdigest()

    # Clear state if a new video is uploaded
    if st.session_state.last_uploaded_video_hash != video_hash:
        st.session_state.last_uploaded_video_hash = video_hash
        st.session_state.metadata = None
        st.session_state.kit_zip = None
        st.session_state.experimental_inference_result_key = None
        st.session_state.experimental_inference_result = None

    is_valid, msg = validate_video(uploaded_video.name, len(video_bytes), config)

    if not is_valid:
        st.error(msg)
    else:
        if st.session_state.kit_zip is None:
            if st.button("Create annotation kit"):
                with st.spinner("Processing video and extracting frames..."):
                    try:
                        metadata, kit_zip = process_and_create_kit(video_bytes, uploaded_video.name, config)
                        st.session_state.metadata = metadata
                        st.session_state.kit_zip = kit_zip
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        else:
            metadata = st.session_state.metadata
            kit_zip = st.session_state.kit_zip

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File name:** {metadata.filename}")
                st.write(f"**File size:** {metadata.size_mb:.2f} MB")
                st.write(f"**Resolution:** {metadata.width}x{metadata.height}")
            with col2:
                st.write(f"**FPS:** {metadata.fps:.2f}")
                st.write(f"**Total Frames:** {metadata.frame_count}")
                st.write(f"**Duration:** {metadata.duration_sec:.2f} seconds")
                st.write(f"**Sampled Frames:** {metadata.sampled_frame_count}")

            st.download_button("Download Annotation Kit", data=kit_zip, file_name="annotation_kit.zip", mime="application/zip")

            with st.expander("Preview Sampled Frames"):
                try:
                    manifest_records = _read_annotation_kit_manifest(kit_zip)
                    with zipfile.ZipFile(BytesIO(kit_zip)) as zf:
                        preview_records = manifest_records[:12]
                        cols = st.columns(4)
                        for idx, record in enumerate(preview_records):
                            img_bytes = zf.read(f"frames/{record['frame_file']}")
                            cols[idx % 4].image(
                                img_bytes,
                                caption=f"{record['frame_file']} ({record['timestamp_seconds']:.2f}s)",
                            )
                        if len(manifest_records) > 12:
                            st.info(f"... and {len(manifest_records) - 12} more frames in the kit.")
                except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                    st.error(f"Unable to preview sampled frames: {exc}")

            if config.experimental_inference.enabled:
                _render_experimental_inference_panel(video_bytes, uploaded_video.name, video_hash, kit_zip)

            st.markdown("---")
            st.subheader("Upload Manual Annotations")
            uploaded_csv = st.file_uploader(
                "Upload your completed manual annotations CSV",
                type=["csv"],
                key=f"manual_csv_{video_hash}"
            )

            if uploaded_csv:
                csv_bytes = uploaded_csv.getvalue()
                with st.spinner("Validating and generating report..."):
                    try:
                        incidents, errors, report_zip, summary = run_analysis(
                            csv_bytes, video_bytes, uploaded_video.name, metadata.width, metadata.height, metadata.fps, metadata.sampled_frame_indices
                        )

                        if errors:
                            st.error("Validation Errors Found:")
                            for e in errors:
                                st.write(f"- {e}")
                        else:
                            st.success("Annotations Validated Successfully!")
                            st.info("Incidents are consolidated from reviewer-provided incident IDs. This is not automatic duplicate removal or ML tracking.")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Manual Observations", summary.total_observations)
                            with col2:
                                st.metric("Grouped Incidents", summary.total_incidents)

                            if incidents:
                                df = pd.DataFrame([inc.model_dump() for inc in incidents])
                                st.dataframe(df)

                                st.download_button("Download Report ZIP", data=report_zip, file_name="roadsense_report.zip", mime="application/zip")

                                st.subheader("Preview Evidence")
                                try:
                                    with zipfile.ZipFile(BytesIO(report_zip)) as zf:
                                        if incidents:
                                            first_inc = incidents[0]
                                            safe_name = "".join(c for c in first_inc.evidence_file if c.isalnum() or c in ('-', '_', '.'))
                                            try:
                                                img_bytes = zf.read(f"evidence/{safe_name}")
                                                st.image(img_bytes, caption=f"Evidence: {first_inc.incident_id}")
                                            except KeyError:
                                                st.error(f"Missing evidence image for {first_inc.incident_id}")

                                        try:
                                            video_data = zf.read("annotated_manual_samples.mp4")
                                            st.video(video_data)
                                        except KeyError:
                                            st.error("Annotated MP4 is missing from report.")
                                except zipfile.BadZipFile:
                                    st.error("Failed to read the generated report ZIP.")

                    except Exception as ex:
                        st.error(f"An error occurred during analysis: {str(ex)}")
            else:
                st.info("Upload a CSV to see results.")
