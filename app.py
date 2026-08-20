import streamlit as st
import pandas as pd
from src.config import load_config
from src.video_io import validate_video, process_and_create_kit
from src.analysis_runner import run_analysis
import zipfile
from io import BytesIO
import hashlib

st.set_page_config(page_title="RoadSense India", layout="wide")

config = load_config()

st.title("RoadSense India")
st.subheader("Step 2: Manual Annotation Baseline")

st.markdown("""
RoadSense currently provides a Manual Annotation Baseline.
It samples video frames and turns human-provided pothole boxes into evidence-backed incident reports.
""")

st.warning("These boxes come from a human-provided CSV. RoadSense has not yet run a trained pothole model, calculated traffic volume, or created a repair priority.")

if "last_uploaded_video_hash" not in st.session_state:
    st.session_state.last_uploaded_video_hash = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "kit_zip" not in st.session_state:
    st.session_state.kit_zip = None

uploaded_video = st.file_uploader("Upload a road video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    video_bytes = uploaded_video.getvalue()
    video_hash = hashlib.sha256(video_bytes).hexdigest()

    # Clear state if a new video is uploaded
    if st.session_state.last_uploaded_video_hash != video_hash:
        st.session_state.last_uploaded_video_hash = video_hash
        st.session_state.metadata = None
        st.session_state.kit_zip = None

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
                with zipfile.ZipFile(BytesIO(kit_zip)) as zf:
                    # extract manifest to get timestamps
                    manifest_data = zf.read("frame_manifest.csv").decode("utf-8").splitlines()
                    timestamps = {}
                    if len(manifest_data) > 1:
                        for line in manifest_data[1:]:
                            parts = line.split(",")
                            if len(parts) >= 3:
                                timestamps[parts[2]] = float(parts[1])

                    # find all jpgs
                    jpgs = [n for n in zf.namelist() if n.startswith("frames/") and n.endswith(".jpg")]
                    jpgs.sort()
                    # show a small grid of max 12 frames
                    preview_jpgs = jpgs[:12]
                    cols = st.columns(4)
                    for idx, jpg_name in enumerate(preview_jpgs):
                        img_bytes = zf.read(jpg_name)
                        base_name = jpg_name.split("/")[-1]
                        ts = timestamps.get(base_name, 0.0)
                        cols[idx % 4].image(img_bytes, caption=f"{base_name} ({ts:.2f}s)")
                    if len(jpgs) > 12:
                        st.info(f"... and {len(jpgs) - 12} more frames in the kit.")

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
