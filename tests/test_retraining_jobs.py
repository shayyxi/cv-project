import sys
import time

import pytest

from app.retraining.jobs import (
    JobBusy,
    JobRunner,
    UnknownJob,
    train_command,
    upload_command,
)


def test_upload_command() -> None:
    assert upload_command({}) == [sys.executable, "-m", "scripts.roboflow_upload"]
    assert upload_command({"dry_run": True})[-1] == "--dry-run"


def test_train_command_defaults_and_options() -> None:
    # No automatic install from the page: the comparison always runs
    # and the replace button decides.
    assert train_command({}) == [
        sys.executable, "-m", "scripts.train_ppe", "--generate",
    ]
    assert "--install" in train_command({"install": True})

    command = train_command(
        {"generate": False, "epochs": 5, "device": "0"}
    )
    assert "--generate" not in command
    assert "--install" not in command
    assert command[-4:] == ["--epochs", "5", "--device", "0"]

    # "auto" means let ultralytics pick: no --device flag.
    assert "--device" not in train_command({"device": "auto"})


def _wait(runner: JobRunner, name: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout

    while time.time() < deadline:
        if not runner.status()[name].running:
            return
        time.sleep(0.1)

    raise AssertionError(f"{name} still running after {timeout}s")


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_job_runs_to_done_and_captures_log(tmp_path) -> None:
    runner = JobRunner(log_dir=tmp_path / "logs", cwd=tmp_path)

    state = runner.start(
        "upload",
        command=_python(
            "import time; print('hello'); time.sleep(0.5); print('bye')"
        ),
    )

    assert state.running
    assert state.log_path is not None
    assert state.log_path.parent == tmp_path / "logs"
    assert runner.busy() == "upload"

    _wait(runner, "upload")

    final = runner.status()["upload"]
    assert final.state == "done"
    assert final.exit_code == 0
    assert final.finished_at is not None
    assert runner.busy() is None

    log = runner.log("upload")
    assert log.splitlines()[0].startswith(sys.executable)  # command header
    assert "hello" in log
    assert log.rstrip().endswith("bye")


def test_failing_job_is_marked_failed(tmp_path) -> None:
    runner = JobRunner(log_dir=tmp_path, cwd=tmp_path)

    runner.start("train", command=_python("import sys; print('boom'); sys.exit(3)"))

    _wait(runner, "train")

    state = runner.status()["train"]
    assert state.state == "failed"
    assert state.exit_code == 3
    assert "boom" in runner.log("train")


def test_only_one_job_at_a_time(tmp_path) -> None:
    runner = JobRunner(log_dir=tmp_path, cwd=tmp_path)

    runner.start("upload", command=_python("import time; time.sleep(2)"))

    with pytest.raises(JobBusy, match="upload"):
        runner.start("train", command=_python("print('never')"))

    _wait(runner, "upload")

    # Free again once the first one finished.
    runner.start("train", command=_python("print('now')"))
    _wait(runner, "train")
    assert runner.status()["train"].state == "done"


def test_unknown_job_and_empty_log(tmp_path) -> None:
    runner = JobRunner(log_dir=tmp_path, cwd=tmp_path)

    with pytest.raises(UnknownJob):
        runner.start("export")

    assert runner.log("upload") == ""
    assert runner.status()["train"].state == "idle"


def test_log_tail_and_carriage_returns(tmp_path) -> None:
    runner = JobRunner(log_dir=tmp_path, cwd=tmp_path)

    runner.start(
        "upload",
        command=_python(
            "import sys\n"
            "for i in range(300): print(f'line {i}')\n"
            "sys.stdout.write('progress 10%\\rprogress 100%\\n')"
        ),
    )

    _wait(runner, "upload")

    tail = runner.log("upload", lines=5).splitlines()
    assert len(tail) == 5
    assert tail[-1] == "progress 100%"
    assert tail[-2] == "progress 10%"
    assert tail[-3] == "line 299"


def test_stop_kills_a_running_job(tmp_path) -> None:
    runner = JobRunner(log_dir=tmp_path, cwd=tmp_path)

    runner.start(
        "train",
        command=_python("import time; print('started'); time.sleep(60)"),
    )
    assert runner.busy() == "train"

    assert runner.stop("train") is True

    state = runner.status()["train"]
    assert state.state == "stopped"
    assert state.finished_at is not None
    assert runner.busy() is None

    # Not running any more: nothing to stop.
    assert runner.stop("train") is False

    # The slot is free again right away.
    runner.start("upload", command=_python("print('ok')"))
    _wait(runner, "upload")
    assert runner.status()["upload"].state == "done"
