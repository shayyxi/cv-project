"""
Background jobs behind the Streamlit buttons.

Each job runs one of the retraining CLI scripts as a subprocess, so a
button does exactly what the command line does. Output goes to a log
file under data/analytics/ui_logs/, which the page tails.
"""

import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Callable

from app.config import BASE_DIR, settings

UI_LOG_DIR = settings.local_analytics_dir / "ui_logs"

ANSI_CODES = re.compile(r"\x1b\[[0-9;]*m")


class JobBusy(RuntimeError):
    """Another job is still running; only one runs at a time."""


class UnknownJob(KeyError):
    """No job definition with that name."""


def upload_command(options: dict) -> list[str]:
    """python -m scripts.roboflow_upload [--dry-run]"""

    command = [sys.executable, "-m", "scripts.roboflow_upload"]

    if options.get("dry_run"):
        command.append("--dry-run")

    return command


def train_command(options: dict) -> list[str]:
    """
    python -m scripts.train_ppe [--generate] [--install]
                                [--epochs N] [--device X]
    """

    command = [sys.executable, "-m", "scripts.train_ppe"]

    if options.get("generate", True):
        command.append("--generate")

    # The page never installs automatically: the comparison always
    # runs, and the replace button on the page makes the decision.
    if options.get("install"):
        command.append("--install")

    epochs = options.get("epochs")

    if epochs:
        command += ["--epochs", str(int(epochs))]

    device = options.get("device")

    if device and str(device).strip().lower() != "auto":
        command += ["--device", str(device).strip()]

    return command


JOBS: dict[str, Callable[[dict], list[str]]] = {
    "upload": upload_command,
    "train": train_command,
}


@dataclass
class JobState:
    name: str
    state: str = "idle"  # idle | running | done | failed | stopped
    command: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    log_path: Path | None = None

    @property
    def running(self) -> bool:
        return self.state == "running"


class JobRunner:
    """
    Starts jobs as subprocesses and tracks them until they exit.

    One instance lives for the whole Streamlit server (see
    scripts/retrain_ui.py), so a running job survives page reloads.
    Only one job runs at a time: training is heavy, and upload and
    train touch the same Roboflow project.
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        cwd: Path | str | None = None,
    ) -> None:
        self._log_dir = Path(log_dir) if log_dir else UI_LOG_DIR
        self._cwd = Path(cwd) if cwd else BASE_DIR

        self._states: dict[str, JobState] = {
            name: JobState(name=name) for name in JOBS
        }
        self._processes: dict[str, subprocess.Popen] = {}
        self._handles: dict[str, IO[bytes]] = {}

        self._lock = threading.Lock()

    def start(
        self,
        name: str,
        options: dict | None = None,
        command: list[str] | None = None,
    ) -> JobState:
        """
        Launch a job. `command` overrides the job's own command (used
        by tests). Raises JobBusy while any job is running.
        """

        if name not in JOBS:
            raise UnknownJob(name)

        with self._lock:
            self._refresh()

            running = self._running_name()

            if running is not None:
                raise JobBusy(
                    f"'{running}' is still running; wait for it to finish."
                )

            command = command or JOBS[name](options or {})

            self._log_dir.mkdir(parents=True, exist_ok=True)

            log_path = (
                self._log_dir
                / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.log"
            )

            handle = log_path.open("wb")
            handle.write((" ".join(command) + "\n\n").encode("utf-8"))
            handle.flush()

            popen_kwargs: dict = {}

            # Own process group, so stop() can take the whole tree
            # (ultralytics dataloader workers included) down with it.
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True

            process = subprocess.Popen(
                command,
                cwd=str(self._cwd),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                **popen_kwargs,
            )

            self._processes[name] = process
            self._handles[name] = handle

            state = JobState(
                name=name,
                state="running",
                command=list(command),
                started_at=datetime.now(),
                log_path=log_path,
            )

            self._states[name] = state

            return state

    def stop(self, name: str, timeout: float = 15.0) -> bool:
        """
        Kill a running job and everything it spawned. Returns False
        when the job was not running. The state becomes "stopped".
        """

        with self._lock:
            self._refresh()

            process = self._processes.get(name)

            if process is None:
                return False

            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                )
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)

            self._finish(name, process.returncode, "stopped")

            return True

    def status(self) -> dict[str, JobState]:
        with self._lock:
            self._refresh()
            return dict(self._states)

    def busy(self) -> str | None:
        """Name of the running job, or None."""

        with self._lock:
            self._refresh()
            return self._running_name()

    def log(self, name: str, lines: int = 200) -> str:
        """
        The last `lines` lines of the job's latest log. Carriage
        returns from progress bars are turned into line breaks.
        """

        return "\n".join(self.log_text(name).splitlines()[-lines:])

    def log_text(self, name: str) -> str:
        """The whole latest log of a job, with progress bars unrolled."""

        state = self.status().get(name)

        if state is None or state.log_path is None:
            return ""

        if not state.log_path.exists():
            return ""

        text = state.log_path.read_text(encoding="utf-8", errors="replace")

        # Progress bars overwrite their line with "\r"; terminal colour
        # codes mean nothing on the page.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return ANSI_CODES.sub("", text)

    def _running_name(self) -> str | None:
        for state in self._states.values():
            if state.running:
                return state.name

        return None

    def _refresh(self) -> None:
        """Flip finished processes to done / failed."""

        for name, process in list(self._processes.items()):
            code = process.poll()

            if code is None:
                continue

            self._finish(name, code, "done" if code == 0 else "failed")

    def _finish(self, name: str, code: int | None, final_state: str) -> None:
        state = self._states[name]
        state.state = final_state
        state.exit_code = code
        state.finished_at = datetime.now()

        handle = self._handles.pop(name, None)

        if handle is not None:
            handle.close()

        self._processes.pop(name, None)
