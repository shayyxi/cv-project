"""
Streamlit page for the retraining loop:

    1. Upload the label queue to Roboflow
    2. Train the PPE model, compare it, replace the deployed one by button
    3. Mark a camera's crop region by clicking on a frame

Start it with:
    streamlit run scripts/retrain_ui.py
or double-click retrain_ui.bat in the project folder.

Each button runs the matching CLI script (scripts.roboflow_upload,
scripts.train_ppe) in the background, shows its progress as a bar,
and can stop it. The last model comparison (deployed vs new, overall
and per class) is shown as tables. Run this on the machine that
should do the training.
"""

import json
import sys
from pathlib import Path

# `streamlit run scripts/retrain_ui.py` puts scripts/ on sys.path, not
# the project root; add the root so `app.*` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.retraining.config import (  # noqa: E402
    HOW_TO_CONFIGURE,
    LABEL_QUEUE_DIR,
    RoboflowSettings,
)
from app.retraining.regions import (  # noqa: E402
    CROP_CONFIG_PATH,
    display_base,
    draw_overlay,
    load_regions,
    normalise_polygon,
    polygon_preview,
    save_region,
    to_full_image,
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # the section explains how to install it
    streamlit_image_coordinates = None
from app.retraining.jobs import JobBusy, JobRunner, JobState  # noqa: E402
from app.retraining.progress import parse_progress  # noqa: E402
from app.retraining.train import (  # noqa: E402
    LATEST_COMPARISON_FILE,
    REGRESSION_TOLERANCE,
    install_weights,
    mark_installed,
    ppe_weights_path,
)
from app.retraining.upload import RoboflowUploadService  # noqa: E402

LOG_LINES = 200


@st.cache_resource
def runner() -> JobRunner:
    """One runner for the whole server, so jobs survive page reloads."""

    return JobRunner()


def pending_crops(config: RoboflowSettings) -> int | None:
    try:
        result = RoboflowUploadService(config=config).upload_new_crops(
            dry_run=True
        )
        return int(result["pending"])
    except Exception:
        return None


def start_job(name: str, options: dict) -> None:
    try:
        state = runner().start(name, options)
    except JobBusy as error:
        st.error(str(error))
        return

    # Show the job that just started in the status panel.
    st.session_state["log_choice"] = name

    st.success(f"Started: {' '.join(state.command)}")
    st.rerun()


def badge(state: JobState) -> str:
    if state.state == "running":
        since = state.started_at.strftime("%H:%M:%S") if state.started_at else ""
        return f":orange[running since {since}]"

    if state.state == "done":
        return ":green[done]"

    if state.state == "failed":
        return f":red[failed (exit code {state.exit_code})]"

    if state.state == "stopped":
        return ":orange[stopped]"

    return ":grey[idle]"


def load_comparison() -> dict | None:
    try:
        return json.loads(LATEST_COMPARISON_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _clamp(value: float | None) -> float:
    return min(max(value if value is not None else 0.0, 0.0), 1.0)


def show_progress(job: str, state: JobState, progress: dict) -> None:
    """
    Two bars: the current step with its own percentage (download,
    extraction, batches of this epoch, ...) and the overall run.
    """

    label = progress["text"]

    if state.state == "stopped":
        label = f"Stopped at: {label}"
    elif state.state == "failed" and not label.startswith("Failed"):
        label = f"Failed at: {label}"

    step = _clamp(progress["phase_fraction"])
    st.progress(step, text=f"Current step: {label} ({step * 100:.0f}%)")

    overall = _clamp(progress["fraction"])

    if job == "train" and progress["epochs"]:
        epochs_done = max(progress["epoch"] - 1, 0)

        if progress["phase"] in ("compare", "done"):
            epochs_done = progress["epochs"]

        st.progress(
            overall,
            text=(
                f"Overall: {overall * 100:.0f}%  |  epochs completed "
                f"{epochs_done} of {progress['epochs']}"
            ),
        )
    else:
        st.progress(overall, text=f"Overall: {overall * 100:.0f}%")


@st.fragment(run_every="2s")
def status_and_log() -> None:
    """Refreshes on its own while a job runs; full rerun when one ends."""

    jobs = runner()
    status = jobs.status()

    busy_now = any(state.running for state in status.values())
    was_busy = st.session_state.get("was_busy", False)
    st.session_state["was_busy"] = busy_now

    if was_busy and not busy_now:
        # A job just finished: re-enable the buttons above.
        st.rerun()

    columns = st.columns(len(status))

    for column, state in zip(columns, status.values()):
        with column:
            st.markdown(f"**{state.name}**: {badge(state)}")

            if state.running:
                if st.button("Stop", key=f"stop-{state.name}"):
                    jobs.stop(state.name)
                    st.rerun()

            elif state.finished_at and state.started_at:
                seconds = int((state.finished_at - state.started_at).total_seconds())
                st.caption(f"took {seconds // 60} min {seconds % 60} s")

    chosen = st.radio(
        "Job",
        list(status),
        horizontal=True,
        key="log_choice",
    )

    state = status[chosen]
    log_text = jobs.log_text(chosen)

    if state.state != "idle":
        show_progress(chosen, state, parse_progress(chosen, log_text))

    # The bar is the view; the raw log is there when something needs
    # a closer look.
    with st.expander("Show log", expanded=False):
        tail = "\n".join(log_text.splitlines()[-LOG_LINES:])
        st.code(tail or "(no run yet)", language="text")


def comparison_section() -> None:
    st.subheader("Last model comparison")

    record = load_comparison()

    if not record:
        st.caption(
            "No comparison yet. It appears after the first Train run "
            "with 'install if not worse' enabled."
        )
        return

    if record.get("installed"):
        outcome = "New model installed"
    elif record.get("better"):
        outcome = "New model is at least as good, not installed yet"
    else:
        outcome = "New model scores lower, not installed"

    st.markdown(
        f"**{outcome}** on the {record.get('split')} split. "
        f"Run `{record.get('run')}`, {record.get('timestamp')}."
    )

    regressed = record.get("regressed_classes") or []

    if regressed:
        st.warning(
            f"mAP50-95 dropped by more than {REGRESSION_TOLERANCE} on: "
            + ", ".join(regressed)
        )

    overall = pd.DataFrame(
        {"deployed": record["deployed"], "new": record["new"]}
    )
    overall["delta"] = overall["new"] - overall["deployed"]

    per_class = pd.DataFrame(record["per_class"]).T

    if not per_class.empty:
        per_class = per_class[["deployed", "new", "delta"]]

    left, right = st.columns(2)

    with left:
        st.markdown("**Overall**")
        st.dataframe(overall.round(4))

    with right:
        st.markdown("**mAP50-95 per class**")
        st.dataframe(per_class.round(4))

    if not per_class.empty:
        chart = per_class[["deployed", "new"]]

        try:
            st.bar_chart(chart, stack=False)
        except TypeError:
            st.bar_chart(chart)

    replace_section(record)


def replace_section(record: dict) -> None:
    """The decision: replace the deployed model with the compared one."""

    flash = st.session_state.pop("install_flash", None)

    if flash:
        st.success(flash)

    if record.get("installed"):
        st.info(
            f"This model was installed on "
            f"{record.get('installed_at') or record.get('timestamp')}. "
            "Restart the pipeline if you have not done so yet."
        )
        return

    new_weights = record.get("new_weights")

    if not new_weights:
        st.caption(
            "This comparison has no weights path recorded; run the "
            "training again to get the replace button."
        )
        return

    if not Path(new_weights).is_file():
        st.error(f"The new model file no longer exists: {new_weights}")
        return

    confirmed = True

    if not record.get("better"):
        st.error(
            "This model scores lower than the deployed one on mAP50-95. "
            "Replacing it would make the pipeline worse."
        )
        confirmed = st.checkbox(
            "I understand it scores lower and want to replace the deployed "
            "model anyway",
            key="confirm_worse",
        )

    busy = runner().busy()

    if st.button(
        "Replace deployed model with this one",
        type="primary",
        disabled=not confirmed or busy is not None,
    ):
        try:
            target, backup = install_weights(Path(new_weights))
        except Exception as error:
            st.error(f"Install failed: {error}")
            return

        mark_installed()

        message = f"Installed -> {target}."

        if backup:
            message += f" Previous weights kept at {backup}."

        message += " Restart the pipeline to load the new model."

        st.session_state["install_flash"] = message
        st.rerun()

    st.caption(
        f"Copies the new best.pt over {ppe_weights_path()} and keeps the "
        "previous file as a backup next to it."
    )


NEW_CAMERA = "new camera id..."

FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")


def latest_raw_frame(camera: str) -> Path | None:
    folder = settings.local_raw_dir / camera

    if not folder.is_dir():
        return None

    frames = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    ]

    return max(frames, key=lambda path: path.stat().st_mtime) if frames else None


def load_frame(source: str, read_bytes) -> tuple[Image.Image, Image.Image, float]:
    """
    (full frame, display-sized copy, scale) for the chosen source,
    kept in the session so clicks do not re-decode a 5 MB frame.
    """

    cache = st.session_state.get("region_frame_cache")

    if cache and cache["source"] == source:
        return cache["image"], cache["base"], cache["scale"]

    import io

    image = Image.open(io.BytesIO(read_bytes()))
    image.load()

    base, scale = display_base(image)

    st.session_state["region_frame_cache"] = {
        "source": source, "image": image, "base": base, "scale": scale,
    }

    return image, base, scale


def region_section() -> None:
    st.subheader("3. Crop region per camera")

    flash = st.session_state.pop("region_flash", None)

    if flash:
        st.success(flash)

    if streamlit_image_coordinates is None:
        st.warning(
            "Marking by clicking needs the streamlit-image-coordinates "
            "package: pip install streamlit-image-coordinates"
        )
        return

    regions = load_regions()

    left, right = st.columns(2)

    with left:
        choice = st.selectbox(
            "Camera", sorted(regions) + [NEW_CAMERA], key="region_camera"
        )

    with right:
        typed = st.text_input(
            "New camera id",
            key="region_new_id",
            disabled=choice != NEW_CAMERA,
        )

    camera = typed.strip() if choice == NEW_CAMERA else choice

    uploaded = st.file_uploader(
        "Frame to mark (jpg or png)",
        type=["jpg", "jpeg", "png"],
        key="region_upload",
    )

    use_raw = st.checkbox(
        "Use the latest raw frame of this camera instead of an upload",
        key="region_use_raw",
    )

    frame = base = None
    scale = 1.0
    source_name = ""

    if use_raw and camera:
        raw = latest_raw_frame(camera)

        if raw is None:
            st.warning(f"No raw frames under {settings.local_raw_dir / camera}.")
        else:
            frame, base, scale = load_frame(str(raw), raw.read_bytes)
            source_name = raw.name

    elif uploaded is not None:
        source = f"upload:{uploaded.name}:{uploaded.size}"
        frame, base, scale = load_frame(source, uploaded.getvalue)
        source_name = uploaded.name

    if frame is None:
        st.caption("Upload a frame, or tick the raw-frame option, to start marking.")
        return

    if not camera:
        st.warning("Type the new camera id first.")
        return

    marks = st.session_state.setdefault("region_marks", {})
    points = marks.setdefault(camera, [])
    saved = regions.get(camera, [])

    st.caption(
        f"{source_name}, {frame.size[0]}x{frame.size[1]} px. Click the corners "
        "of the area the model should look at; two clicks make a rectangle, "
        "more clicks a polygon. Blue = saved region, red = your new points."
    )

    overlay, scale = draw_overlay(frame, saved, points, base=base)

    click = streamlit_image_coordinates(
        overlay, key=f"region-canvas-{camera}", image_format="JPEG"
    )

    if click and click != st.session_state.get("region_last_click"):
        st.session_state["region_last_click"] = click

        # The component reports on-screen pixels and the on-screen size
        # of the picture; the picture itself is the display-sized copy.
        shown_width = click.get("width") or overlay.size[0]
        css_scale = overlay.size[0] / shown_width

        points.append(
            to_full_image(click["x"] * css_scale, click["y"] * css_scale, scale)
        )
        st.rerun()

    undo, clear, reuse = st.columns(3)

    if undo.button("Undo last point", disabled=not points):
        points.pop()
        st.rerun()

    if clear.button("Clear points", disabled=not points):
        marks[camera] = []
        st.rerun()

    if reuse.button("Load saved polygon into points", disabled=not saved):
        marks[camera] = list(saved)
        st.rerun()

    if not points:
        return

    table, preview = st.columns([1, 2])

    with table:
        st.dataframe(
            pd.DataFrame(points, columns=["x", "y"], index=range(1, len(points) + 1))
        )

    if len(points) < 2:
        return

    pending = normalise_polygon(points)

    with preview:
        # Preview on the display-sized copy: same shape, far cheaper.
        small = [(x / scale, y / scale) for x, y in pending]
        st.image(
            polygon_preview(base, small),
            caption="What the model will see (black = outside the region)",
            width=600,
        )

    st.code(
        f'"{camera}":\n  polygon:\n'
        + "\n".join(f"    - [{x}, {y}]" for x, y in pending),
        language="yaml",
    )

    if st.button("Save region to config", type="primary"):
        written = save_region(camera, points, image_size=frame.size)
        marks[camera] = []
        st.session_state["region_flash"] = (
            f"Saved {len(written)} points for camera {camera} to "
            f"{CROP_CONFIG_PATH}. The pipeline uses it from its next frame."
        )
        st.rerun()


# Same fonts and palette as the PPE Camera Monitor dashboard. Colours live
# in .streamlit/config.toml; fonts and the finer details are set here.
THEME_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap");

html, body, [class*="st-"], button, input, textarea, select {
    font-family: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
/* Streamlit draws its arrows and other icons with an icon font; keep it. */
[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
}
h1, h2, h3, code, pre, [data-testid="stMetricValue"] {
    font-family: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace !important;
}
h1 { font-weight: 600; letter-spacing: -0.02em; }
h2, h3 { font-weight: 600; }
[data-testid="stHeader"] { background: #0a0d11; }
[data-testid="stCaptionContainer"] { color: #7f8896; }
[data-testid="stExpander"], [data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 4px;
}
.stButton > button, .stDownloadButton > button { border-radius: 4px; font-weight: 500; }
hr { border-color: rgba(255, 255, 255, 0.10); }
/* Hide the hover link icon next to headings. */
[data-testid="stHeaderActionElements"] { display: none; }
</style>
"""


# Light palette of the same dashboard, laid over the dark theme.
LIGHT_CSS = """
<style>
.stApp, [data-testid="stHeader"] { background: #eceef2; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp span:not([data-testid="stIconMaterial"]), .stApp summary { color: #141920; }
[data-testid="stCaptionContainer"] { color: #7c8493; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
input:not([type="checkbox"]), textarea, [data-testid="stFileUploaderDropzone"] {
    background: #fbfbfc !important; color: #141920 !important;
}
/* The toggle: grey track when off, accent blue when on, white knob. */
[data-baseweb="checkbox"] > div:first-child { background: #c4c9d2 !important; }
[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
    background: #2a78d6 !important;
}
[data-baseweb="checkbox"] > div:first-child > div { background: #ffffff !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: #fbfbfc; color: #141920; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { background: #f2f3f6 !important; color: #141920; }
[data-testid="stExpander"], [data-testid="stDataFrame"], hr {
    border-color: rgba(20, 25, 32, 0.11);
}
.stButton > button[kind="primary"] { color: #ffffff; }
.stButton > button[kind="primary"] span { color: #ffffff; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="PPE retraining", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    # Read before the toggle is drawn so the first paint already matches.
    if not st.session_state.get("dark_mode", True):
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    _, toggle_column = st.columns([6, 1])
    toggle_column.toggle("Dark mode", value=True, key="dark_mode")

    config = RoboflowSettings.from_env()
    busy = runner().busy()

    st.title("PPE model retraining")
    st.caption(
        f"Label queue: {LABEL_QUEUE_DIR}  |  "
        f"deployed weights: {ppe_weights_path()}"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("1. Upload label queue to Roboflow")

        pending = pending_crops(config)
        st.write(
            "Crops waiting for upload: "
            + (f"**{pending}**" if pending is not None else "unknown")
        )

        if not config.configured:
            st.warning(f"Roboflow is not configured; {HOW_TO_CONFIGURE}.")

        dry_run = st.checkbox("Dry run (only count, send nothing)", value=False)

        if st.button(
            "Upload",
            type="primary",
            disabled=busy is not None or (not config.configured and not dry_run),
        ):
            start_job("upload", {"dry_run": dry_run})

        st.caption(
            "Sends each crop once with its pre-labels. Review the batch "
            "in Roboflow and add it to the dataset before training."
        )

    with right:
        st.subheader("2. Train and deploy")

        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
        device_choice = st.selectbox(
            "Device",
            ["gpu", "cpu"],
            index=0,
            help=(
                "Where training runs. gpu: the first NVIDIA GPU (fast, needs "
                "one). cpu: the processor (slow, works anywhere)."
            ),
        )
        # The training script calls the first GPU "0".
        device = "0" if device_choice == "gpu" else "cpu"
        generate = st.checkbox(
            "Generate a new dataset version from the current labels",
            value=True,
        )

        if st.button(
            "Train",
            type="primary",
            disabled=busy is not None or not config.configured,
        ):
            start_job(
                "train",
                {
                    "epochs": int(epochs),
                    "device": device,
                    "generate": generate,
                },
            )

        st.caption(
            "Fine-tunes from the deployed best.pt and compares both on the "
            "test split. Nothing is replaced automatically: the numbers and "
            "the replace button appear at the bottom of the page."
        )

    if busy is not None:
        st.info(f"'{busy}' is running. Use Stop below to abort it.")

    st.divider()

    status_and_log()

    st.divider()

    comparison_section()

    st.divider()

    region_section()


main()
