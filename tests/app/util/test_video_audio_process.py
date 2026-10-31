"""Process contracts exercised with real lightweight Python children, without FFmpeg."""

import subprocess
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from invokeai.app.util import video_audio
from invokeai.app.util.process_tree import kill_process_tree


@pytest.fixture
def child_process(monkeypatch):
    children = []
    launches = []
    killed = []

    def install(script):
        def spawn(command, **kwargs):
            launches.append((command, kwargs))
            child = subprocess.Popen([sys.executable, "-c", script], **kwargs)
            children.append(child)
            return child

        # Replace only this module's binding, never stdlib Popen used by process-tree cleanup.
        monkeypatch.setattr(video_audio, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": spawn}))
        monkeypatch.setattr(video_audio, "_ffmpeg_exe", lambda: "test-ffmpeg")

    def kill(child):
        killed.append(child)
        kill_process_tree(child)

    monkeypatch.setattr(video_audio, "kill_process_tree", kill)
    yield install, launches, children, killed
    for child in children:
        if child.poll() is None:
            kill_process_tree(child)


@pytest.mark.parametrize("capture", [False, True])
def test_stdout_capture_and_default_keep_stdin_isolated_and_stderr_bounded(child_process, capture):
    install, launches, children, killed = child_process
    install(
        "import sys\n"
        "assert sys.stdin.buffer.read() == b''\n"
        "sys.stdout.buffer.write(b';FFMETADATA1\\n' + b'x' * (2 * 1024 * 1024))\n"
        "sys.stderr.buffer.write(b'prefix' + b'e' * 8192)\n"
        "sys.exit(7)\n"
    )
    kwargs = {"capture_stdout": True} if capture else {}
    result = video_audio.run_ffmpeg_process(["-version"], 20, **kwargs)
    command, options = launches[0]
    assert command == ["test-ffmpeg", "-y", "-loglevel", "error", "-version"]
    assert options["stdin"] == subprocess.DEVNULL
    if capture:
        assert options["stdout"] != subprocess.PIPE
        assert options["stdout"].closed
    else:
        assert options["stdout"] == subprocess.DEVNULL
    assert options["stderr"].closed
    assert result.args == command
    assert result.returncode == children[0].returncode == 7
    assert result.stdout == (b";FFMETADATA1\n" + b"x" * (2 * 1024 * 1024) if capture else b"")
    assert result.stderr == b"e" * 8192
    assert not killed


def test_cancel_between_decode_and_probe_starts_no_probe(child_process, monkeypatch, tmp_path):
    install, launches, _, _ = child_process
    install("raise AssertionError('must not launch')")
    canceled = Event()
    real_run = video_audio.run_ffmpeg_process

    def run(args, **kwargs):
        if "ffmetadata" not in args:
            canceled.set()
            return subprocess.CompletedProcess(args, 1, b"", b"decode failed")
        assert kwargs["is_canceled"] == canceled.is_set
        return real_run(args, **kwargs)

    monkeypatch.setattr(video_audio, "run_ffmpeg_process", run)
    monkeypatch.setattr(video_audio.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(video_audio.CanceledException):
        video_audio.extract_audio_pcm(Path("silent.mp4"), is_canceled=canceled.is_set)
    assert not launches
    assert not list(tmp_path.glob("invokeai_audio_extract_*"))


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_running_probe_is_killed_reaped_and_output_cleaned(child_process, monkeypatch, tmp_path, stop):
    install, launches, children, killed = child_process
    install("from threading import Event; Event().wait()")
    real_run = video_audio.run_ffmpeg_process
    started = Event()

    def is_canceled():
        # The first check occurs before Popen. The next is after its bounded wait.
        if children:
            started.set()
        return stop == "cancel" and started.is_set()

    def run(args, **kwargs):
        if "ffmetadata" not in args:
            return subprocess.CompletedProcess(args, 1, b"", b"decode failed")
        assert kwargs["timeout"] == 60
        assert kwargs["is_canceled"] is is_canceled
        assert kwargs["capture_stdout"] is True
        # Expire on the first poll, without waiting out the production deadline.
        return real_run(args, **{**kwargs, "timeout": 0 if stop == "timeout" else 20})

    monkeypatch.setattr(video_audio, "run_ffmpeg_process", run)
    monkeypatch.setattr(video_audio.tempfile, "tempdir", str(tmp_path))
    error = video_audio.CanceledException if stop == "cancel" else video_audio.AudioExtractionError
    with pytest.raises(error) as caught:
        video_audio.extract_audio_pcm(Path("silent.mp4"), is_canceled=is_canceled)
    if stop == "timeout":
        assert "timed out probing audio in silent.mp4" in str(caught.value)
        assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    assert started.is_set()
    assert len(children) == 1
    assert killed == children
    assert children[0].returncode is not None
    assert children[0].wait(timeout=0) == children[0].returncode
    assert launches[0][1]["stdin"] == subprocess.DEVNULL
    assert launches[0][1]["stdout"].closed
    assert launches[0][1]["stderr"].closed
    assert not list(tmp_path.glob("invokeai_audio_extract_*"))
