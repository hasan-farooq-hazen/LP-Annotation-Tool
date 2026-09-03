"""Streamlit UI for the License Plate Annotation Tool."""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import get_args

import cv2
import streamlit as st
from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel

from hazen_license_plate_annotation_tool.pipeline import (
    StageResult,
    extract_plates,
    extract_video_frames,
    image_paths,
    make_zip_bytes,
    reduce_similar_frames,
    unpack_image_uploads,
)


APP_NAME = "Hazen License Plate Annotation Tool"
FPS_OPTIONS = [5, 10, 15, 20]
SIMILARITY_LEVELS = {
    "Gentle — only very close matches": 3,
    "Balanced — removes moderate repetition": 6,
    "Strong — recommended": 10,
}
DETECTOR_MODELS = list(get_args(PlateDetectorModel))
OCR_MODELS = list(get_args(OcrModel))
DEFAULT_DETECTOR = "yolo-v9-s-608-license-plate-end2end"
DEFAULT_OCR = "cct-s-v2-global-model"


st.set_page_config(
    page_title=APP_NAME,
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      :root {
        --ink: #17212b;
        --muted: #607080;
        --accent: #0c7c86;
        --accent-soft: #e8f5f4;
        --warm: #f3a847;
        --line: #dce5e7;
      }
      .stApp { background: #f7faf9; color: var(--ink); }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: #102a32; }
      [data-testid="stSidebar"] * { color: #edf7f5; }
      [data-testid="stSidebar"] .stButton button {
        min-height: 2.5rem;
        justify-content: flex-start;
        padding: .55rem .75rem;
        border-radius: 9px;
        font-weight: 600;
        transition: background .15s ease, border-color .15s ease;
      }
      [data-testid="stSidebar"] .stButton button * {
        color: inherit !important;
      }
      [data-testid="stSidebar"] .stButton button[kind="secondary"] {
        color: #d8e8e7;
        background: #173840;
        border: 1px solid #31535a;
      }
      [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
        color: #ffffff;
        background: #21464f;
        border-color: #4f747b;
      }
      [data-testid="stSidebar"] .stButton button[kind="primary"] {
        color: #ffffff;
        background: #0c7c86;
        border-color: #26a0a7;
      }
      [data-testid="stSidebar"] details {
        padding: 0;
        border: 0;
        background: transparent;
      }
      [data-testid="stSidebar"] details summary {
        min-height: 2.5rem;
        padding: .55rem .7rem;
        border: 1px solid #31535a;
        border-radius: 9px;
        background: transparent;
        color: #b9cece;
        font-weight: 600;
      }
      [data-testid="stSidebar"] details summary:hover,
      [data-testid="stSidebar"] details[open] summary {
        color: #ffffff;
        background: #173840;
        border-color: #4f747b;
      }
      [data-testid="stSidebar"] details[open] summary {
        margin-bottom: .4rem;
      }
      [data-testid="stSidebar"] details [data-testid="stVerticalBlock"] {
        gap: .45rem;
      }
      .app-title { font-size: 2.2rem; line-height: 1.1; font-weight: 760; margin: 0; }
      .app-subtitle { color: var(--muted); font-size: 1rem; max-width: 760px; margin-top: .6rem; }
      .step-card {
        min-height: 128px; padding: 1rem 1.05rem; border: 1px solid var(--line);
        border-radius: 14px; background: white; box-shadow: 0 4px 18px rgba(25,52,58,.04);
      }
      .step-number {
        display: inline-flex; width: 28px; height: 28px; align-items: center;
        justify-content: center; border-radius: 8px; background: var(--accent-soft);
        color: var(--accent); font-weight: 800; margin-bottom: .5rem;
      }
      .step-card strong { display: block; margin-bottom: .25rem; }
      .step-card span { color: var(--muted); font-size: .9rem; }
      .section-note {
        border-left: 4px solid var(--warm); background: #fff8ec; padding: .75rem 1rem;
        border-radius: 0 10px 10px 0; color: #61461f; margin: .5rem 0 1.25rem;
      }
      .quick-start {
        padding: .8rem 1rem; margin: .25rem 0 1rem; border-radius: 10px;
        background: var(--accent-soft); color: #18545a; font-weight: 600;
      }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid var(--line); border-radius: 12px; padding: .7rem 1rem;
      }
      .stButton > button[kind="primary"] {
        background: var(--accent); border-color: var(--accent); border-radius: 10px;
      }
      .stDownloadButton > button { border-radius: 10px; }
      h1, h2, h3 { letter-spacing: -.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _workspace() -> Path:
    if "workspace_handle" not in st.session_state:
        st.session_state.workspace_handle = tempfile.TemporaryDirectory(prefix="plate_tool_")
        st.session_state.artifacts = {}
        st.session_state.downloads = {}
    return Path(st.session_state.workspace_handle.name)


def _new_directory(label: str) -> Path:
    path = _workspace() / f"{label}_{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True)
    return path


def _save_video(upload, scope: str) -> Path:
    payload = upload.getvalue()
    identity = hashlib.sha1(payload).hexdigest()
    cache_key = f"video_upload_{scope}"
    cached = st.session_state.get(cache_key)
    if cached and cached[0] == identity and Path(cached[1]).is_file():
        return Path(cached[1])
    suffix = Path(upload.name).suffix.lower() or ".mp4"
    path = _new_directory("video_input") / f"uploaded_video{suffix}"
    path.write_bytes(payload)
    st.session_state[cache_key] = (identity, str(path))
    return path


def _video_details(path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0, 0.0, 0
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    return duration, fps, frames


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _video_controls(scope: str):
    upload = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv", "m4v"],
        key=f"video_{scope}",
        help="MP4 is the most reliable format. Your video is processed in this session.",
    )
    path = None
    duration = fps = 0.0
    if upload is not None:
        path = _save_video(upload, scope)
        duration, fps, _ = _video_details(path)
        if duration:
            st.caption(
                f"Video ready · {_format_duration(duration)} long · {fps:.1f} source FPS · "
                f"{upload.size / (1024 * 1024):.1f} MB"
            )

    col1, col2 = st.columns(2)
    with col1:
        target_fps = st.select_slider(
            "Frames per second",
            options=FPS_OPTIONS,
            value=10,
            key=f"fps_{scope}",
            help="Higher values capture more detail but take longer and create larger downloads.",
        )
    with col2:
        trim_enabled = st.checkbox(
            "Process only part of the video",
            key=f"trim_enabled_{scope}",
            disabled=not bool(duration),
        )

    start, end = 0.0, None
    if trim_enabled and duration > 0:
        selected = st.slider(
            "Choose the section to process (seconds)",
            min_value=0.0,
            max_value=max(1.0, float(duration)),
            value=(0.0, float(duration)),
            step=1.0,
            key=f"trim_range_{scope}",
        )
        start, end = selected
        st.caption(f"Selected section: {_format_duration(start)} to {_format_duration(end)}")

    jpeg_quality = 92
    with st.expander("Advanced frame settings"):
        jpeg_quality = st.slider(
            "Image quality",
            min_value=70,
            max_value=100,
            value=100,
            key=f"quality_{scope}",
            help="Maximum quality preserves the most detail for plate recognition.",
        )
    return upload, path, target_fps, jpeg_quality, start, end


def _similarity_controls(scope: str) -> tuple[int, int]:
    level = st.radio(
        "How aggressively should similar frames be removed?",
        options=list(SIMILARITY_LEVELS),
        index=2,
        key=f"similarity_{scope}",
        help="Strong is recommended for reducing repeated traffic and security-camera frames.",
    )
    keep = st.selectbox(
        "How many frames should be kept from each similar group?",
        options=[1, 2, 3],
        index=1,
        key=f"keep_{scope}",
        help="The clearest frame is kept first.",
    )
    return SIMILARITY_LEVELS[level], keep


def _plate_controls(scope: str) -> tuple[float, str, str]:
    confidence = st.slider(
        "Plate detection confidence",
        min_value=0.20,
        max_value=0.90,
        value=0.70,
        step=0.05,
        key=f"confidence_{scope}",
        help="Raise this if non-plate objects are being selected. Lower it if real plates are missed.",
    )
    detector = DEFAULT_DETECTOR if DEFAULT_DETECTOR in DETECTOR_MODELS else DETECTOR_MODELS[0]
    ocr = DEFAULT_OCR if DEFAULT_OCR in OCR_MODELS else OCR_MODELS[0]
    with st.expander("Advanced recognition settings"):
        detector = st.selectbox(
            "Detection model",
            DETECTOR_MODELS,
            index=DETECTOR_MODELS.index(detector),
            key=f"detector_{scope}",
        )
        ocr = st.selectbox(
            "Reading model",
            OCR_MODELS,
            index=OCR_MODELS.index(ocr),
            key=f"ocr_{scope}",
        )
    return confidence, detector, ocr


@st.cache_resource(show_spinner=False)
def _get_alpr(detector: str, ocr: str, confidence: float) -> ALPR:
    return ALPR(
        detector_model=detector,
        detector_conf_thresh=confidence,
        ocr_model=ocr,
    )


def _progress_callback(label: str):
    bar = st.progress(0.0, text=label)

    def update(current: int, total: int, text: str) -> None:
        ratio = min(1.0, current / total) if total else 0.0
        bar.progress(ratio, text=text)

    return bar, update


def _remember_artifact(name: str, result: StageResult) -> None:
    st.session_state.artifacts[name] = result
    st.session_state.downloads[name] = make_zip_bytes(result.output_dir)


def _show_result(
    name: str,
    title: str,
    download_name: str,
    download_label: str = "Download results (.zip)",
) -> None:
    result = st.session_state.artifacts.get(name)
    if result is None:
        return
    st.success(f"{title} completed")
    first, second, third = st.columns(3)
    first.metric("Input", f"{result.input_count:,}")
    second.metric("Created", f"{result.output_count:,}")
    third.metric("Skipped", f"{result.skipped_count:,}")
    if result.error_count:
        st.warning(f"{result.error_count:,} item(s) could not be processed. Details are in manifest.csv.")
    st.download_button(
        download_label,
        data=st.session_state.downloads[name],
        file_name=download_name,
        mime="application/zip",
        key=f"download_{name}",
        use_container_width=True,
    )


def _uploaded_image_directory(files, scope: str) -> Path:
    directory = _new_directory(f"{scope}_input")
    uploads = [(file.name, file.getvalue()) for file in files]
    written = unpack_image_uploads(uploads, directory)
    if not written:
        raise ValueError("Please upload JPG/PNG images or a ZIP containing images.")
    return directory


def _choose_image_source(scope: str, artifact_names: list[tuple[str, str]]):
    available = [item for item in artifact_names if item[0] in st.session_state.artifacts]
    labels = [label for _, label in available]
    labels.append("Upload images or a ZIP")
    choice = st.radio(
        "Where should the images come from?",
        labels,
        key=f"source_{scope}",
        horizontal=True,
    )
    files = None
    source = None
    if choice == "Upload images or a ZIP":
        files = st.file_uploader(
            "Choose JPG/PNG images or ZIP files",
            type=["jpg", "jpeg", "png", "zip"],
            accept_multiple_files=True,
            key=f"images_{scope}",
        )
    else:
        artifact_name = next(name for name, label in available if label == choice)
        source = st.session_state.artifacts[artifact_name].output_dir
        st.caption(f"{len(image_paths(source)):,} images are ready from the earlier step.")
    return source, files


def _run_frame_stage(video_path, fps, quality, start, end) -> StageResult:
    output = _new_directory("frames")
    bar, callback = _progress_callback("Preparing frame extraction")
    result = extract_video_frames(video_path, output, fps, quality, start, end, callback)
    bar.empty()
    _remember_artifact("frames", result)
    return result


def _header() -> None:
    st.markdown(f'<div class="app-title">{APP_NAME}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Turn vehicle video into a clean, reviewable set of '
        'license-plate crops. Run one tool at a time or let the guided workflow handle everything.</div>',
        unsafe_allow_html=True,
    )
    st.write("")


_workspace()
_header()

with st.sidebar:
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "Guided workflow"

    st.markdown("### Main workflow")
    st.caption("Recommended for most users")
    if st.button(
        "★ Guided workflow",
        key="nav_guided",
        type="primary" if st.session_state.active_mode == "Guided workflow" else "secondary",
        use_container_width=True,
    ):
        st.session_state.active_mode = "Guided workflow"

    with st.expander("Individual steps"):
        st.caption("Choose one task to run on its own")
        individual_modes = [
            ("Video → frames", "1 · Video to frames"),
            ("Reduce similar frames", "2 · Reduce similar frames"),
            ("Extract plates", "3 · Extract license plates"),
        ]
        for index, (button_label, individual_mode) in enumerate(individual_modes, start=1):
            if st.button(
                button_label,
                key=f"nav_individual_{index}",
                type=(
                    "primary"
                    if st.session_state.active_mode == individual_mode
                    else "secondary"
                ),
                use_container_width=True,
            ):
                st.session_state.active_mode = individual_mode

    mode = st.session_state.active_mode
    st.markdown("---")
    st.caption("WORK COMPLETED THIS SESSION")
    for artifact, label in (
        ("frames", "Frames"),
        ("reduced", "Reduced set"),
        ("plates", "Plate crops"),
    ):
        result = st.session_state.artifacts.get(artifact)
        st.write(f"{'✓' if result else '○'} {label}" + (f" · {result.output_count:,}" if result else ""))
    st.caption("Files stay available while this browser session remains open.")


if mode == "Guided workflow":
    st.subheader("From video to plate crops")
    columns = st.columns(3)
    cards = [
        ("1", "Create frames", "Choose 5–20 FPS and optionally trim the video."),
        ("2", "Remove repeats", "Keep the clearest examples from similar groups."),
        ("3", "Extract plates", "Crop each plate and use OCR to name it."),
    ]
    for column, (number, title, text) in zip(columns, cards):
        column.markdown(
            f'<div class="step-card"><div class="step-number">{number}</div>'
            f'<strong>{title}</strong><span>{text}</span></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown(
        '<div class="quick-start">Quick start: upload your video and press “Process video '
        'and extract plates.” The recommended settings are already selected.</div>',
        unsafe_allow_html=True,
    )
    upload, video_path, fps, quality, start, end = _video_controls("guided")
    with st.expander("Similarity settings · recommended defaults selected", expanded=False):
        distance, keep = _similarity_controls("guided")
    with st.expander("Plate recognition settings · recommended defaults selected", expanded=False):
        confidence, detector, ocr = _plate_controls("guided")

    if st.button(
        "Process video and extract plates",
        type="primary",
        disabled=upload is None,
        use_container_width=True,
    ):
        try:
            with st.status("Processing your video", expanded=True) as status:
                st.write("1 of 3 · Creating frames")
                frame_result = _run_frame_stage(video_path, fps, quality, start, end)
                st.write(f"Created {frame_result.output_count:,} frames")

                st.write("2 of 3 · Selecting the clearest unique frames")
                reduced_output = _new_directory("reduced")
                bar, callback = _progress_callback("Comparing frames")
                reduced_result = reduce_similar_frames(
                    frame_result.output_dir, reduced_output, distance, keep, callback
                )
                bar.empty()
                _remember_artifact("reduced", reduced_result)
                st.write(f"Kept {reduced_result.output_count:,} frames")

                st.write("3 of 3 · Finding and reading license plates")
                plate_output = _new_directory("plates")
                bar, callback = _progress_callback("Loading recognition models")
                alpr = _get_alpr(detector, ocr, confidence)
                plate_result = extract_plates(reduced_output, plate_output, alpr, callback)
                bar.empty()
                _remember_artifact("plates", plate_result)
                status.update(label="All three steps completed", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Processing stopped: {exc}")

    if "plates" in st.session_state.artifacts:
        st.markdown("### Your results")
        _show_result(
            "plates",
            "Plate extraction",
            "license_plate_crops.zip",
            "Download plate crops (.zip)",
        )
        if "reduced" in st.session_state.downloads:
            st.download_button(
                "Download reduced frames (.zip)",
                st.session_state.downloads["reduced"],
                "reduced_frames.zip",
                "application/zip",
                key="download_guided_reduced",
                use_container_width=True,
            )

elif mode == "1 · Video to frames":
    st.subheader("Create frames from a video")
    st.markdown(
        '<div class="section-note">Use this when you only need still images. Plate detection is not '
        'run in this step, so every sampled frame is delivered.</div>',
        unsafe_allow_html=True,
    )
    upload, video_path, fps, quality, start, end = _video_controls("frames_only")
    if st.button(
        "Create frames", type="primary", disabled=upload is None, use_container_width=True
    ):
        try:
            _run_frame_stage(video_path, fps, quality, start, end)
        except Exception as exc:
            st.error(f"The video could not be processed: {exc}")
    _show_result("frames", "Frame extraction", "video_frames.zip")

elif mode == "2 · Reduce similar frames":
    st.subheader("Reduce similar frames")
    st.markdown(
        '<div class="section-note">The clearest image in each similar group is kept. This tool works '
        'independently—you can upload images made by any system.</div>',
        unsafe_allow_html=True,
    )
    source, files = _choose_image_source("reduce", [("frames", "Use frames from step 1")])
    distance, keep = _similarity_controls("reduce")
    ready = source is not None or bool(files)
    if st.button(
        "Reduce similar frames", type="primary", disabled=not ready, use_container_width=True
    ):
        try:
            input_dir = source or _uploaded_image_directory(files, "reduce")
            output = _new_directory("reduced")
            bar, callback = _progress_callback("Comparing frames")
            result = reduce_similar_frames(input_dir, output, distance, keep, callback)
            bar.empty()
            _remember_artifact("reduced", result)
        except Exception as exc:
            st.error(f"The images could not be processed: {exc}")
    _show_result("reduced", "Similarity reduction", "reduced_frames.zip")

else:
    st.subheader("Extract license plates")
    st.markdown(
        '<div class="section-note">Each detected plate is saved as a separate image. Recognized '
        'plate text is used as the filename; unreadable plates are still kept.</div>',
        unsafe_allow_html=True,
    )
    source, files = _choose_image_source(
        "plates",
        [("reduced", "Use reduced frames from step 2"), ("frames", "Use frames from step 1")],
    )
    confidence, detector, ocr = _plate_controls("plates")
    ready = source is not None or bool(files)
    if st.button(
        "Extract license plates", type="primary", disabled=not ready, use_container_width=True
    ):
        try:
            input_dir = source or _uploaded_image_directory(files, "plates")
            output = _new_directory("plates")
            bar, callback = _progress_callback("Loading recognition models")
            alpr = _get_alpr(detector, ocr, confidence)
            result = extract_plates(input_dir, output, alpr, callback)
            bar.empty()
            _remember_artifact("plates", result)
        except Exception as exc:
            st.error(f"Plate extraction could not be completed: {exc}")
    _show_result("plates", "Plate extraction", "license_plate_crops.zip")
