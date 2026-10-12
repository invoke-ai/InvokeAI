"""No-audio detection must use container structure, not FFmpeg's error prose.

Tiny real media fixtures exercise both PCM paths. Error-wording substitution is
deliberate fault injection, not a claim that the bundled FFmpeg changed wording.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock

import numpy as np
import pytest

from invokeai.app.util import video_audio


@pytest.fixture
def media(tmp_path: Path, monkeypatch):
    silent = tmp_path / "silent.mp4"
    audio = tmp_path / "audio.wav"
    ffmpeg = video_audio._ffmpeg_exe()
    for args in (
        ["-f", "lavfi", "-i", "color=size=16x16:rate=1", "-t", "1", "-c:v", "libx264", str(silent)],
        ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "0.1", str(audio)],
    ):
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", *args], check=True, capture_output=True, timeout=20)
    monkeypatch.setattr(video_audio.tempfile, "tempdir", str(tmp_path))
    return silent, audio


@pytest.mark.parametrize("float_pcm", [False, True])
@pytest.mark.parametrize("stderr", [b"No output audio track", b"Aucune piste audio"])
def test_silent_clip_does_not_depend_on_error_wording(media, monkeypatch, float_pcm, stderr):
    silent, _ = media
    real_run = subprocess.run

    def run(command, **kwargs):
        proc = real_run(command, **kwargs)
        if "-acodec" in command:
            assert proc.returncode != 0  # Real FFmpeg rejects a zero-stream PCM output.
            return subprocess.CompletedProcess(command, proc.returncode, proc.stdout, stderr)
        return proc

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    assert video_audio.extract_audio_pcm(silent, float_pcm=float_pcm) is None
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


@pytest.mark.parametrize("float_pcm", [False, True])
@pytest.mark.parametrize("input_kind", ["corrupt", "missing"])
def test_invalid_input_is_not_silence_even_if_error_mentions_no_stream(media, monkeypatch, float_pcm, input_kind):
    silent, _ = media
    invalid = silent.parent / f"{input_kind}.mp4"
    if input_kind == "corrupt":
        invalid.write_bytes(b"not a media container")
    real_run = subprocess.run

    def run(command, **kwargs):
        proc = real_run(command, **kwargs)
        if "-acodec" in command:
            assert proc.returncode != 0
            return subprocess.CompletedProcess(command, proc.returncode, proc.stdout, b"does not contain any stream")
        return proc

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError):
        video_audio.extract_audio_pcm(invalid, float_pcm=float_pcm)
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


@pytest.mark.parametrize("float_pcm", [False, True])
def test_existing_audio_decode_error_is_not_silence(media, monkeypatch, float_pcm):
    _, audio = media
    real_run = subprocess.run

    def run(command, **kwargs):
        if "-acodec" in command:
            return subprocess.CompletedProcess(command, 1, b"", b"decoder does not contain any stream")
        return real_run(command, **kwargs)

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError, match="decoder does not contain any stream"):
        video_audio.extract_audio_pcm(audio, float_pcm=float_pcm)
    assert not list(audio.parent.glob("invokeai_audio_extract_*"))


@pytest.mark.parametrize("float_pcm", [False, True])
def test_successful_decode_does_not_add_a_structural_probe(media, monkeypatch, float_pcm):
    _, audio = media
    real_run = subprocess.run
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    result = video_audio.extract_audio_pcm(audio, float_pcm=float_pcm)
    assert result is not None
    pcm, rate = result
    assert rate == 44100
    assert pcm.shape == (2, 4410)
    assert len(calls) == (2 if float_pcm else 1)  # Float PCM already probes its rate.
    assert not any("ffmetadata" in call for call in calls)


@pytest.mark.parametrize(
    "returncode,stdout",
    [(1, b";FFMETADATA1\n"), (0, b""), (0, b"unexpected output"), (0, b";FFMETADATA2\n")],
)
def test_failed_or_unrecognized_probe_preserves_decode_error(media, monkeypatch, returncode, stdout):
    silent, _ = media

    def run(command, **kwargs):
        if "ffmetadata" in command:
            return subprocess.CompletedProcess(command, returncode, stdout, b"probe failed")
        return subprocess.CompletedProcess(command, 1, b"", b"original decode error")

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError, match="original decode error"):
        video_audio.extract_audio_pcm(silent)
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


def test_structural_probe_timeout_is_bounded_and_cleans_output(media, monkeypatch):
    silent, _ = media

    def run(command, **kwargs):
        if "ffmetadata" in command:
            assert kwargs["timeout"] == 60
            raise subprocess.TimeoutExpired(command, 60)
        return subprocess.CompletedProcess(command, 1, b"", b"original decode error")

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError, match="timed out probing audio"):
        video_audio.extract_audio_pcm(silent)
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


def test_structural_probe_launch_failure_preserves_decode_error(media, monkeypatch):
    silent, _ = media

    def run(command, **kwargs):
        if "ffmetadata" in command:
            raise OSError("probe binary unavailable")
        return subprocess.CompletedProcess(command, 1, b"", b"original decode error")

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError, match="original decode error"):
        video_audio.extract_audio_pcm(silent)


def test_extraction_timeout_does_not_start_a_probe(media, monkeypatch):
    silent, _ = media

    def run(command, **kwargs):
        assert "ffmetadata" not in command
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError, match="timed out extracting audio"):
        video_audio.extract_audio_pcm(silent)
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


@pytest.mark.parametrize("float_pcm", [False, True])
def test_damaged_audio_payload_is_not_silence(media, monkeypatch, float_pcm):
    silent, audio = media
    damaged = silent.parent / "damaged.m4a"
    subprocess.run(
        [video_audio._ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(audio), "-c:a", "aac", str(damaged)],
        check=True,
        capture_output=True,
        timeout=20,
    )
    encoded = damaged.read_bytes()
    marker = encoded.index(b"mdat")
    atom_size = int.from_bytes(encoded[marker - 4 : marker], "big")
    payload_end = marker - 4 + atom_size
    damaged.write_bytes(encoded[: marker + 4] + bytes(payload_end - marker - 4) + encoded[payload_end:])
    real_run = subprocess.run
    results = []

    def run(command, **kwargs):
        proc = real_run(command, **kwargs)
        results.append((command, proc))
        return proc

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    with pytest.raises(video_audio.AudioExtractionError):
        video_audio.extract_audio_pcm(damaged, float_pcm=float_pcm)
    assert results[0][1].returncode != 0  # Real AAC decoding, not an injected failure.
    if len(results) > 1:  # Baseline has no probe; the fixed path must recognize the stream.
        assert "ffmetadata" in results[1][0]
        assert results[1][1].returncode == 0
        assert b"[STREAM]" in results[1][1].stdout.splitlines()
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


def test_silent_clip_metadata_cannot_spoof_stream_section(media, monkeypatch):
    silent, _ = media
    tagged = silent.parent / "tagged.mp4"
    subprocess.run(
        [
            video_audio._ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-c",
            "copy",
            "-metadata",
            "comment=prefix\n[STREAM]",
            str(tagged),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    # This exact marker really occurs without metadata suppression. A trailing
    # suffix would escape the newline after it and make this regression vacuous.
    metadata = subprocess.run(
        [
            video_audio._ffmpeg_exe(),
            "-loglevel",
            "error",
            "-i",
            str(tagged),
            "-map",
            "0:a:0?",
            "-vn",
            "-sn",
            "-dn",
            "-map_metadata",
            "0",
            "-map_chapters",
            "-1",
            "-c",
            "copy",
            "-t",
            "0",
            "-f",
            "ffmetadata",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert b"[STREAM]" in metadata.stdout.splitlines()
    real_run = subprocess.run

    def run(command, **kwargs):
        proc = real_run(command, **kwargs)
        if "-acodec" in command:
            return subprocess.CompletedProcess(command, proc.returncode, proc.stdout, b"No output audio track")
        return proc

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    assert video_audio.extract_audio_pcm(tagged) is None
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))


@pytest.fixture(params=["subtitle_only", "video_subtitle", "video_subtitle_attachment"])
def non_audio_streams(request, tmp_path: Path, monkeypatch):
    subtitle = tmp_path / "captions.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:00,500\nNo audio here\n", encoding="utf-8")
    output = tmp_path / f"{request.param}.mkv"
    command = [video_audio._ffmpeg_exe(), "-y", "-loglevel", "error"]
    if request.param == "subtitle_only":
        command += ["-i", str(subtitle), "-map", "0:s", "-c:s", "srt"]
    else:
        command += [
            "-f",
            "lavfi",
            "-i",
            "color=size=16x16:rate=2",
            "-i",
            str(subtitle),
            "-map",
            "0:v",
            "-map",
            "1:s",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:s",
            "srt",
        ]
        if request.param.endswith("attachment"):
            attachment = tmp_path / "data.bin"
            attachment.write_bytes(b"application-owned attachment; not an audio stream")
            command += ["-attach", str(attachment), "-metadata:s:t:0", "mimetype=application/octet-stream"]
    subprocess.run([*command, str(output)], check=True, capture_output=True, timeout=30)
    monkeypatch.setattr(video_audio.tempfile, "tempdir", str(tmp_path))
    return output


@pytest.mark.parametrize("float_pcm", [False, True])
def test_subtitles_and_attachments_are_not_audio(non_audio_streams, monkeypatch, float_pcm):
    real_run = subprocess.run
    probes = []

    def run(command, **kwargs):
        proc = real_run(command, **kwargs)
        if "ffmetadata" in command:
            probes.append(proc)
        if "-acodec" in command:
            assert proc.returncode != 0  # Real decoder failure precedes prose substitution.
            return subprocess.CompletedProcess(command, proc.returncode, proc.stdout, b"Aucune piste audio")
        return proc

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    assert video_audio.extract_audio_pcm(non_audio_streams, float_pcm=float_pcm) is None
    assert len(probes) == 1
    assert probes[0].returncode == 0
    assert b"[STREAM]" not in probes[0].stdout.splitlines()
    assert not list(non_audio_streams.parent.glob("invokeai_audio_extract_*"))


def test_stream_marker_alone_does_not_identify_audio(non_audio_streams):
    # Real subtitle selection emits the same marker. Audio-only selection is
    # therefore part of the classifier's contract, not merely an optimization.
    command = [
        video_audio._ffmpeg_exe(),
        "-loglevel",
        "error",
        "-i",
        str(non_audio_streams),
        "-map",
        "0:s:0?",
        "-vn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c",
        "copy",
        "-t",
        "0",
        "-f",
        "ffmetadata",
        "pipe:1",
    ]
    subtitles = subprocess.run(command, check=True, capture_output=True, timeout=30)
    audio_only = subprocess.run(
        ["0:a:0?" if argument == "0:s:0?" else argument for argument in command],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert b"[STREAM]" in subtitles.stdout.splitlines()
    assert b"[STREAM]" not in audio_only.stdout.splitlines()


@pytest.mark.parametrize("float_pcm", [False, True])
def test_failed_extractions_do_not_remove_concurrent_audio_outputs(media, monkeypatch, float_pcm):
    silent, first = media
    second = silent.parent / "second.wav"
    corrupt = silent.parent / "corrupt.mp4"
    missing = silent.parent / "missing.mp4"
    corrupt.write_bytes(b"not a media container")
    real_run = subprocess.run
    ffmpeg = video_audio._ffmpeg_exe()
    real_run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000",
            "-t",
            "0.1",
            str(second),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    successful = {first, second}
    unsuccessful = {silent, corrupt, missing}
    expected = {}
    for path in successful:
        # Decode a separate raw reference, not through the helper under test.
        codec, format_name, dtype = (
            ("pcm_f32le", "f32le", np.float32) if float_pcm else ("pcm_s16le", "s16le", np.int16)
        )
        reference = real_run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-ac",
                "2",
                "-acodec",
                codec,
                "-f",
                format_name,
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
        samples = np.frombuffer(reference.stdout, dtype=dtype).reshape(-1, 2).T
        expected[path] = samples if float_pcm else samples.astype(np.float32) / 32768.0

    # All five calls allocate their outputs before any decoder can complete.
    # Both successful decoders finish writing before failed decodes can return.
    # Successes then retain their output until every failed call has returned
    # through finally. This exercises cleanup, without sleeps or racing
    # the host scheduler, while real FFmpeg does the decoding and probing.
    allocated = Barrier(5, timeout=40)
    written = {path: Event() for path in successful}
    cleaned = {path: Event() for path in unsuccessful}
    lock = Lock()
    outputs = {}
    errors = {}
    probes = []

    def run(command, **kwargs):
        if "-i" not in command:
            return real_run(command, **kwargs)
        path = Path(command[command.index("-i") + 1])
        is_decode = "-acodec" in command
        if is_decode:
            with lock:
                outputs[path] = Path(command[-1])
            allocated.wait()
        proc = real_run(command, **kwargs)
        if is_decode and path in successful:
            assert proc.returncode == 0
            written[path].set()
            for event in cleaned.values():
                assert event.wait(40), "A failed extraction did not finish cleanup"
            assert outputs[path].is_file(), "Another extraction removed this call's PCM"
        elif is_decode:
            assert proc.returncode != 0  # Genuine invalid/no-audio inputs.
            for event in written.values():
                assert event.wait(40), "A successful decoder did not finish writing"
            with lock:
                errors[path] = proc.stderr.decode("utf-8", errors="replace").strip()[-500:]
        elif "ffmetadata" in command:
            with lock:
                probes.append(path)
        return proc

    def extract(path):
        try:
            try:
                return video_audio.extract_audio_pcm(path, float_pcm=float_pcm)
            except video_audio.AudioExtractionError as error:
                return error
        finally:
            if path in cleaned:
                cleaned[path].set()

    monkeypatch.setattr(video_audio.subprocess, "run", run)
    inputs = [first, second, silent, corrupt, missing]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = dict(zip(inputs, pool.map(extract, inputs), strict=True))
    assert len(outputs) == len(set(outputs.values())) == 5
    for path, rate in ((first, 44100), (second, 48000)):
        result = results[path]
        assert isinstance(result, tuple)
        pcm, actual_rate = result
        assert actual_rate == rate
        assert pcm.shape == (2, rate // 10)
        assert np.array_equal(pcm, expected[path])
    assert results[silent] is None
    for path in (corrupt, missing):
        assert isinstance(results[path], video_audio.AudioExtractionError)
        assert errors[path]
        assert errors[path] in str(results[path])
    assert set(probes).issubset(unsuccessful)
    assert not list(silent.parent.glob("invokeai_audio_extract_*"))
