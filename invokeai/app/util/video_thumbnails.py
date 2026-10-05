"""Video frame/probe helpers used by the video file store, upload router, and video nodes.

Decoding runs in a short-lived child process (``video_decode_worker.py``) with a hard
timeout. Files reaching these helpers are user uploads, and both decode backends can
hang indefinitely on crafted or malformed containers (cv2 wheels historically, ffmpeg in
degenerate cases). An in-process hang would tie up the FastAPI request worker that
called us — repeated crafted uploads could exhaust the worker pool — and a hung thread
cannot be killed from Python, so process isolation is the only reliable bound.
The parent explicitly terminates the worker and its descendants when the timeout
expires, so a hostile file costs at most ``timeout`` seconds and cannot leak a stuck
FFmpeg process.
"""

import io
import json
import math
import os
import queue
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional

import numpy as np
import psutil
from PIL import Image

from invokeai.app.services.session_processor.session_processor_common import CanceledException

# Generous — a healthy decode of a single frame or of container metadata takes well
# under a second even for large files. The timeout exists to bound adversarial or hung
# decodes, not to police slow ones.
VIDEO_DECODE_TIMEOUT_SECONDS = 30.0
MAX_DECODED_FRAME_RECORD_BYTES = 256 * 1024 * 1024
MAX_DECODE_STDERR_BYTES = 16 * 1024
# Resident-memory kill threshold for a decode worker *tree* (the worker plus its
# ffmpeg child). Sized for a legal max-size decode, not a typical one: a
# MAX_VIDEO_FRAME_PIXELS RGB frame is 192 MiB, the worker's emit path holds ~4
# frame-sized copies at once (~768 MiB), and the ffmpeg child keeps its own output
# frame plus YUV reference frames (several hundred MiB more) — so a 1 GiB bound would
# kill decodes of videos the frame-size validators explicitly accept. This is an
# anti-runaway bound, not a budget. Keep in sync with WORKER_MEMORY_HEADROOM_BYTES in
# video_decode_worker.py (the worker-side address-space bound).
MAX_VIDEO_DECODE_WORKER_RSS_BYTES = 3 * 1024 * 1024 * 1024
# A bound this size doesn't need 50 ms granularity; each poll walks the process tree
# via psutil, so a coarser period keeps sustained streaming decodes cheap.
WORKER_MEMORY_POLL_SECONDS = 0.25
# Upper bound on decoded frame size (~8K video). Decoder-reported dimensions are
# untrusted metadata: a tiny crafted container can claim absurd dimensions whose full
# frame is only allocated at decode time. probe_video rejects such files before the
# upload path persists them, and the worker refuses to decode frames from them.
# Keep in sync with MAX_FRAME_PIXELS in video_decode_worker.py.
MAX_VIDEO_FRAME_PIXELS = 64 * 1024 * 1024
MAX_CONCURRENT_VIDEO_DECODERS = 2
_VIDEO_DECODER_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_VIDEO_DECODERS)
_VIDEO_STREAM_SLOTS = threading.BoundedSemaphore(1)

_WORKER_PATH = Path(__file__).parent / "video_decode_worker.py"


class VideoDecodeTimeoutError(TimeoutError):
    """The decode worker did not finish within its time budget.

    A timeout is contention (or an adversarial file), not evidence the video is
    undecodable — callers that gate acceptance on decodability must treat it as
    inconclusive rather than as a failed decode.
    """


def get_video_thumbnail_name(video_name: str) -> str:
    """Given a video file name (e.g. <uuid>.mp4), returns the matching thumbnail name (e.g. <uuid>.webp)."""
    return os.path.splitext(video_name)[0] + ".webp"


def _worker_command(*args: str) -> list[str]:
    """Command line for one decode-worker invocation (patchable in tests).

    The worker is run by file path rather than ``-m`` so the child process doesn't
    import the invokeai package (and transitively torch) just to decode a frame.
    """
    return [sys.executable, str(_WORKER_PATH), *args]


def _spawn_worker(*args: str, **kwargs: Any) -> subprocess.Popen[Any]:
    """Starts a worker in an independently killable process group on POSIX."""
    if os.name != "nt":
        kwargs["start_new_session"] = True
    return subprocess.Popen(_worker_command(*args), **kwargs)


def _is_process_running(pid: int) -> bool:
    """Returns whether a process exists and is not a zombie."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Kills a worker and every descendant it spawned."""
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        parent = psutil.Process(proc.pid)
        processes = parent.children(recursive=True)
        processes.append(parent)
        for process in processes:
            try:
                process.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(processes, timeout=5)
    except psutil.Error:
        try:
            proc.kill()
        except OSError:
            pass


def _worker_tree_rss(proc: subprocess.Popen[Any]) -> int:
    """Returns resident memory used by the worker and its decoder children."""
    try:
        worker = psutil.Process(proc.pid)
        processes = [worker, *worker.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.Error:
            pass
    return total


def _start_worker_memory_monitor(
    proc: subprocess.Popen[Any],
) -> tuple[threading.Event, threading.Event, threading.Thread]:
    """Kills a decoder process tree when its cross-platform RSS bound is exceeded."""
    stopped = threading.Event()
    exceeded = threading.Event()

    def monitor() -> None:
        while not stopped.is_set():
            if proc.poll() is not None:
                return
            if _worker_tree_rss(proc) > MAX_VIDEO_DECODE_WORKER_RSS_BYTES:
                exceeded.set()
                _terminate_process_tree(proc)
                return
            stopped.wait(WORKER_MEMORY_POLL_SECONDS)

    thread = threading.Thread(target=monitor, name="video-memory-monitor", daemon=True)
    thread.start()
    return stopped, exceeded, thread


def _run_worker_unbounded(args: list[str], timeout: float, raise_on_timeout: bool = False) -> Optional[dict[str, Any]]:
    """Runs the decode worker; returns its parsed JSON output, or None on failure or timeout.

    With ``raise_on_timeout``, a timeout raises VideoDecodeTimeoutError instead of
    returning None, letting callers distinguish "could not decode" from "ran out of
    time on a loaded machine".
    """
    monitor_stop: threading.Event | None = None
    monitor: threading.Thread | None = None
    try:
        proc = _spawn_worker(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None
    try:
        monitor_stop, _memory_exceeded, monitor = _start_worker_memory_monitor(proc)
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _terminate_process_tree(proc)
        proc.communicate()
        if raise_on_timeout:
            raise VideoDecodeTimeoutError(f"Video decode worker timed out after {timeout}s") from error
        return None
    except Exception:
        # An unexpected failure (e.g. OSError from communicate()) must not leak the
        # worker tree: the finally below stops the RSS-monitor backstop, so nothing
        # else would ever reap a still-running worker and its ffmpeg child.
        _terminate_process_tree(proc)
        return None
    finally:
        if monitor_stop is not None and monitor is not None:
            monitor_stop.set()
            monitor.join(timeout=1)
    if proc.returncode != 0:
        return None
    try:
        result = json.loads(stdout)
    except ValueError:
        return None
    return result if isinstance(result, dict) else None


def _run_worker(args: list[str], timeout: float, raise_on_timeout: bool = False) -> Optional[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    if not _VIDEO_DECODER_SLOTS.acquire(timeout=timeout):
        if raise_on_timeout:
            raise VideoDecodeTimeoutError(f"Video decode worker timed out after {timeout}s")
        return None
    try:
        remaining = max(0.0, deadline - time.monotonic())
        return _run_worker_unbounded(args, remaining, raise_on_timeout)
    finally:
        _VIDEO_DECODER_SLOTS.release()


def _iter_video_frames_unbounded(
    video_path: Path,
    timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS,
    is_canceled: Optional[Callable[[], bool]] = None,
    first_frame_timeout: Optional[float] = None,
) -> Iterator[np.ndarray]:
    """Streams decoded frames from an isolated worker with bounded memory and wait time.

    ``timeout`` bounds decoder *inactivity*: it is restarted after every frame, so a long
    video is not killed for being long. ``first_frame_timeout`` overrides that budget for
    the first frame only, letting a caller that already spent part of the budget waiting
    for capacity charge that wait against the same deadline instead of granting a fresh one.
    """
    proc = _spawn_worker(
        "stream",
        str(video_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None or proc.stderr is None:
        _terminate_process_tree(proc)
        raise RuntimeError("Unable to open video decoder output stream")
    memory_monitor_stop, _memory_exceeded, memory_monitor = _start_worker_memory_monitor(proc)

    results: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    stopped = threading.Event()
    stderr_tail = bytearray()

    def read_stderr() -> str:
        return bytes(stderr_tail).decode(errors="replace").strip()

    def drain_stderr() -> None:
        while chunk := proc.stderr.read(4096):
            stderr_tail.extend(chunk)
            if len(stderr_tail) > MAX_DECODE_STDERR_BYTES:
                del stderr_tail[:-MAX_DECODE_STDERR_BYTES]

    def read_exactly(size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                if remaining == size:
                    raise EOFError
                raise OSError("Truncated frame record from video decoder")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def put_result(result: tuple[str, object]) -> None:
        while not stopped.is_set():
            try:
                results.put(result, timeout=0.1)
                return
            except queue.Full:
                continue

    def read_frames() -> None:
        try:
            while not stopped.is_set():
                record_size = struct.unpack(">Q", read_exactly(8))[0]
                if record_size > MAX_DECODED_FRAME_RECORD_BYTES:
                    raise ValueError(f"Decoded frame record exceeds {MAX_DECODED_FRAME_RECORD_BYTES} bytes")
                payload = read_exactly(record_size)
                put_result(("frame", np.load(io.BytesIO(payload), allow_pickle=False)))
        except (EOFError, ValueError, OSError) as error:
            put_result(("done", error))

    reader = threading.Thread(target=read_frames, name="video-frame-reader", daemon=True)
    stderr_reader = threading.Thread(target=drain_stderr, name="video-stderr-reader", daemon=True)
    reader.start()
    stderr_reader.start()
    deadline = time.monotonic() + (timeout if first_frame_timeout is None else first_frame_timeout)
    try:
        while True:
            if is_canceled is not None and is_canceled():
                raise CanceledException
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out decoding frames from {video_path}")
            try:
                kind, value = results.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if kind == "frame":
                if not isinstance(value, np.ndarray):
                    raise ValueError(f"Decoder returned an invalid frame for {video_path}")
                yield value
                deadline = time.monotonic() + timeout
                continue
            try:
                return_code = proc.wait(timeout=min(1, timeout))
            except subprocess.TimeoutExpired as error:
                _terminate_process_tree(proc)
                stderr_reader.join(timeout=1)
                detail = read_stderr()
                message = f"Timed out waiting for video decoder worker for {video_path}"
                raise TimeoutError(f"{message}: {detail}" if detail else message) from error
            if return_code != 0:
                stderr_reader.join(timeout=1)
                detail = read_stderr()
                message = f"Unable to decode video at {video_path}"
                raise ValueError(f"{message}: {detail}" if detail else message) from value
            return
    finally:
        memory_monitor_stop.set()
        memory_monitor.join(timeout=1)
        stopped.set()
        if proc.poll() is None:
            _terminate_process_tree(proc)
        proc.stdout.close()
        proc.wait()
        reader.join(timeout=1)
        stderr_reader.join(timeout=1)
        proc.stderr.close()


def iter_video_frames(
    video_path: Path,
    timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> Iterator[np.ndarray]:
    acquired: list[threading.BoundedSemaphore] = []
    capacity_deadline = time.monotonic() + timeout
    try:
        for slot in (_VIDEO_STREAM_SLOTS, _VIDEO_DECODER_SLOTS):
            while True:
                if is_canceled is not None and is_canceled():
                    raise CanceledException
                remaining = capacity_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting to decode frames from {video_path}")
                if slot.acquire(timeout=min(0.1, remaining)):
                    acquired.append(slot)
                    break
        # Charge the capacity wait against the same deadline as the first frame, the way
        # _run_worker does. Handing the decoder a fresh full timeout here would let a
        # caller that waited just under `timeout` for a slot block for nearly 2 * timeout
        # before failing — twice the bound the callers (upload probing, node decodes)
        # believe they are enforcing. Later frames still get a full `timeout` each: after
        # the first frame the budget is an inactivity bound, not a queueing one.
        yield from _iter_video_frames_unbounded(
            video_path,
            timeout,
            is_canceled,
            first_frame_timeout=max(0.0, capacity_deadline - time.monotonic()),
        )
    finally:
        for slot in reversed(acquired):
            slot.release()


def extract_video_frame(
    video_path: Path,
    frame_index: int = 0,
    timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS,
    raise_on_timeout: bool = False,
) -> Optional[Image.Image]:
    """Extracts a single frame from a video file as a PIL Image. Returns None on failure or timeout.

    With ``raise_on_timeout``, a timeout raises VideoDecodeTimeoutError instead of
    returning None.
    """
    fd, tmp_name = tempfile.mkstemp(prefix="invokeai_frame_", suffix=".png")
    os.close(fd)
    try:
        result = _run_worker(["frame", str(video_path), str(frame_index), tmp_name], timeout, raise_on_timeout)
        if result is None:
            return None
        with Image.open(tmp_name) as image:
            image.load()
        return image
    except VideoDecodeTimeoutError:
        raise
    except Exception:
        return None
    finally:
        Path(tmp_name).unlink(missing_ok=True)


# Where in a clip the gallery thumbnail search starts. Frame 0 is a poor representative:
# generated videos commonly fade in from black or start on a conditioning frame, and an
# audio-only upload wrapped in a synthesized waveform track (see video_ingest.py) renders its
# first frame from a near-empty audio window — an all-black tile. About a second in, capped at
# the clip's midpoint so short clips still resolve to a real frame, is far more representative.
THUMBNAIL_FRAME_TARGET_SECONDS = 1.0
# Used when the container reports no usable fps; matches the video models' native rate and the
# synthesized waveform track's rate.
THUMBNAIL_FRAME_FALLBACK_FPS = 24.0

# Where to look when the frame at the first rung decodes but is empty (see
# FrameScore.informative): the long fade-in or title-sequence case, which no fixed offset can
# cover because it scales with the runtime. Fractions of the duration; a candidate must land
# meaningfully *later* than the frame just rejected, which is what walks the ladder forward
# past a title sequence rather than back into it. More fractions than MAX_DEEPER_SEEK_ATTEMPTS
# on purpose: the early ones are unusable on a short clip and get skipped.
DEEPER_SEEK_FRACTIONS: tuple[float, ...] = (0.1, 0.35, 0.6)
MIN_DEEPER_SEEK_GAP_SECONDS = 1.0
# Ceiling on the deeper rungs. Each attempt costs a decode-worker spawn, and the upload
# path runs this inside the request. With the opening rung and the frame-0 walk-back the
# search is at most MAX_DEEPER_SEEK_ATTEMPTS + 2 decodes.
MAX_DEEPER_SEEK_ATTEMPTS = 2

# A candidate frame passes three gates read off one luma histogram: it is not flat, not
# dark, and not mostly one level. The first two thresholds are ported from PhotoMapAI, which
# calibrated them against a labelled corpus of synthetic-but-realistic frames and real
# encodes; the third was added after the first two let real title cards through (see
# FRAME_FLAT_FRACTION_CEILING). Erring towards rejection is deliberate throughout: a frame
# wrongly called empty only costs another decode, and the best-scoring frame is returned
# either way; a frame wrongly kept is the black thumbnail this whole search exists to avoid.
#
# Shannon entropy of the luma histogram, in bits, below which a frame is flat: a black
# screen, a solid slate, a fade. Junk in the calibration corpus measured 0.0-0.8 bits (pure
# black 0.0, dense end credits ~0.65); real content starts around 1.1 even when almost
# entirely dark — a night skyline, fireworks against black, an overcast snowfield — because
# a dark *scene* has tone everywhere while a slate is one level.
FRAME_ENTROPY_FLOOR = 0.75

# Entropy measures flatness, not darkness, and the two come apart at the bottom of the
# range. A photograph dimmed to a peak luma of 2/255 — black to any viewer — still measures
# ~1.45 bits, because the dither spread across a handful of levels carries information;
# black carrying nothing but sensor noise measures ~1.27 after an x264 round trip. Both
# clear the entropy floor, and the floor cannot be raised to catch them: the dimmest real
# content in the calibration corpus measures 1.07. So darkness is asked about separately,
# as a high quantile of the luma histogram: "are there any genuinely bright pixels", not
# "is the average bright", so fireworks on a night sky survive. A quantile rather than the
# maximum, so one stuck pixel, a timecode burn-in or a logo cannot vouch for an otherwise
# black frame.
FRAME_HIGHLIGHT_QUANTILE = 0.999
# Between the junk band (a 1% fade measures 2, a 2% fade 5, black with sensor noise 7) and
# the dimmest frame accepted from a real library (32). Low rather than high because a
# rejected frame falls through to the entropy ranking, where grain over black outranks
# genuinely dim content: a frame has to be darker than anything with recoverable content
# before it is put at that risk.
FRAME_LUMA_FLOOR = 10.0

# Fraction of pixels within FRAME_FLAT_BAND luma levels of the most common level above
# which a frame is a card: titles, a logo or a caption on a solid background. Synthetic
# title cards measure 0.2-0.3 bits and fail the entropy floor, but real encoded ones do
# not reliably: three title cards from a real library measured 0.74, 0.80 and 0.94 bits,
# straddling the floor, because x264 ringing around anti-aliased text spreads a few percent
# of pixels across most of the histogram. What they have in common is the background: 93-96%
# of their pixels sit on one level, while no accepted frame from the same library exceeded
# 0.68 (a portrait on white seamless, a pillarboxed clip) and ordinary content sits below
# 0.1. The band absorbs codec noise on the background; the ceiling sits in the middle of the
# gap. A product shot or a slide with a large plain background can fail this gate, at the
# cost of the extra decodes only: every rung of such a clip fails alike, and the
# highest-entropy one is returned.
FRAME_FLAT_BAND = 2
FRAME_FLAT_FRACTION_CEILING = 0.9


class FrameScore(NamedTuple):
    """The three histogram measures of a thumbnail candidate. See the gate constants above."""

    entropy: float
    highlight_luma: float
    flat_fraction: float

    @classmethod
    def measure(cls, frame: Image.Image) -> "FrameScore":
        """Scores ``frame`` from one luma histogram; an empty frame measures as kept."""
        histogram = frame.convert("L").histogram()
        total = sum(histogram)
        if not total:
            return cls(math.inf, 255.0, 0.0)
        entropy = -sum((count / total) * math.log2(count / total) for count in histogram if count)
        # Whole pixels, floored at one: a float headroom makes the boundary arbitrary
        # (1000 * (1 - 0.999) is 1.0000000000000009), and the floor keeps a frame of very
        # few pixels measurable. Read from white downwards, so it costs 256 steps, not a sort.
        headroom = max(1, int(total * (1.0 - FRAME_HIGHLIGHT_QUANTILE)))
        seen = 0
        highlight = 0.0
        for level in range(255, -1, -1):
            seen += histogram[level]
            if seen >= headroom:
                highlight = float(level)
                break
        mode = max(range(256), key=histogram.__getitem__)
        flat = sum(histogram[max(0, mode - FRAME_FLAT_BAND) : mode + FRAME_FLAT_BAND + 1]) / total
        return cls(entropy, highlight, flat)

    @property
    def gates_passed(self) -> int:
        return (
            int(self.entropy >= FRAME_ENTROPY_FLOOR)
            + int(self.highlight_luma >= FRAME_LUMA_FLOOR)
            + int(self.flat_fraction < FRAME_FLAT_FRACTION_CEILING)
        )

    @property
    def informative(self) -> bool:
        """All three gates have to hold: flat frames are slates, dark ones fades, mostly-one-level ones cards."""
        return self.gates_passed == 3

    @property
    def rank(self) -> tuple[int, float]:
        """Ordering among frames that failed: fewest gates failed, then entropy.

        Gates first because entropy alone ranks grain over black (~1.3 bits, fails the luma
        gate) above a legible title card (~0.8 bits, fails only the flat gate); when nothing
        better is reachable the card is the thumbnail a person would pick.
        """
        return (self.gates_passed, self.entropy)


def thumbnail_frame_candidates(duration: Optional[float], fps: Optional[float]) -> list[int]:
    """The frame indices a gallery thumbnail is tried from, in order.

    The opening rung is roughly THUMBNAIL_FRAME_TARGET_SECONDS in, capped at the clip's
    midpoint; then up to MAX_DEEPER_SEEK_ATTEMPTS rungs at DEEPER_SEEK_FRACTIONS of the
    duration, each meaningfully later than the last; then frame 0 as the walk-back, which
    doubles as the fallback for containers whose metadata overstates the decodable range.
    Returns ``[0]`` when the duration is unknown or degenerate — callers without metadata
    keep first-frame behavior. Both inputs are untrusted container metadata, so a
    non-finite value degrades to the safe answer rather than raising.
    """
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return [0]
    effective_fps = fps if fps is not None and math.isfinite(fps) and fps > 0 else THUMBNAIL_FRAME_FALLBACK_FPS
    seconds = [min(THUMBNAIL_FRAME_TARGET_SECONDS, duration / 2)]
    for fraction in DEEPER_SEEK_FRACTIONS:
        if len(seconds) > MAX_DEEPER_SEEK_ATTEMPTS:
            break
        candidate = duration * fraction
        if candidate >= seconds[-1] + MIN_DEEPER_SEEK_GAP_SECONDS:
            seconds.append(candidate)
    candidates: list[int] = []
    # Finite inputs can still multiply to infinity; such a rung cannot be decoded, so it is
    # dropped rather than raised on.
    for index in [*(int(offset * effective_fps) for offset in seconds if math.isfinite(offset * effective_fps)), 0]:
        if index not in candidates:
            candidates.append(index)
    return candidates


def extract_representative_video_frame(
    video_path: Path,
    duration: Optional[float] = None,
    fps: Optional[float] = None,
    timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS,
    raise_on_timeout: bool = False,
) -> Optional[Image.Image]:
    """Extracts the thumbnail frame: the first informative one on the seek ladder, else the best.

    Walks thumbnail_frame_candidates in order and returns the first frame FrameScore calls
    informative. If every reachable frame is empty — a clip that really is all dark, or the
    waveform track wrapping an audio upload — the best-ranked frame seen is returned, so the
    video still gets a thumbnail rather than a gallery placeholder; the waveform case thereby
    lands on the busiest window of the track.

    An opening rung with no decodable frame means the metadata overstates the range or the
    file is damaged there; the deeper rungs are later still, so the search goes straight to
    the frame-0 walk-back rather than spending the budget proving each one empty.

    The whole search shares one budget of two ``timeout``s, the same worst case the
    two-attempt version had (a slow decode *failure* followed by a full fallback decode).
    A timeout, or an exhausted budget, ends the search: it is contention or an adversarial
    file, not evidence about the index, and another rung would hold a request worker for
    another full budget. If a frame was already decoded it is returned — a dark thumbnail
    beats none — otherwise the VideoDecodeTimeoutError propagates with ``raise_on_timeout``
    and the call returns None without it, as ``extract_video_frame`` would.
    """
    queue = thumbnail_frame_candidates(duration, fps)
    opening = queue[0]
    best: Optional[tuple[tuple[int, float], Image.Image]] = None
    deadline = time.monotonic() + 2 * timeout
    try:
        while queue:
            frame_index = queue.pop(0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoDecodeTimeoutError(f"Thumbnail search for {video_path} exhausted its {2 * timeout}s budget")
            frame = extract_video_frame(
                video_path, frame_index=frame_index, timeout=min(timeout, remaining), raise_on_timeout=True
            )
            if frame is None:
                if frame_index == opening and queue:
                    queue = [queue[-1]]
                continue
            score = FrameScore.measure(frame)
            if score.informative:
                return frame
            if best is None or score.rank > best[0]:
                best = (score.rank, frame)
            # Frames can be up to MAX_VIDEO_FRAME_PIXELS; don't hold a rejected one across
            # the next decode.
            del frame
    except VideoDecodeTimeoutError:
        if best is None:
            if raise_on_timeout:
                raise
            return None
    return best[1] if best is not None else None


def probe_video_with_codec(
    video_path: Path, timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS
) -> tuple[int, int, float, Optional[float], Optional[str]]:
    """Returns (width, height, duration_seconds, fps_or_none, codec_or_none) for a video file.

    Raises FileNotFoundError if the file cannot be read — including when the decode
    times out, since a file we cannot probe within the bound is treated as unreadable —
    or when the decoder reports metadata no sane video has (non-positive or over-limit
    dimensions, non-finite or negative duration). Decoder-reported values are untrusted:
    they come from the uploaded container, and the upload path persists them and sizes
    thumbnail decoding by them. A non-finite or non-positive fps is coerced to None
    (unknown) rather than rejected, matching the decoder's own unknown-fps behavior.
    """
    result = _run_worker(["probe", str(video_path)], timeout)
    if result is None:
        raise FileNotFoundError(f"Unable to open video at {video_path}")
    try:
        width = int(result["width"])
        height = int(result["height"])
        duration = float(result["duration"])
        fps_raw = result.get("fps")
        fps: Optional[float] = float(fps_raw) if fps_raw else None
        codec_raw = result.get("codec")
        codec = str(codec_raw).lower() if codec_raw else None
    except (KeyError, TypeError, ValueError, OverflowError) as e:
        raise FileNotFoundError(f"Unable to open video at {video_path}") from e
    if width <= 0 or height <= 0 or width * height > MAX_VIDEO_FRAME_PIXELS:
        raise FileNotFoundError(f"Video at {video_path} reports invalid dimensions {width}x{height}")
    if not math.isfinite(duration) or duration < 0:
        raise FileNotFoundError(f"Video at {video_path} reports an invalid duration {duration}")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        fps = None
    return width, height, duration, fps, codec


def probe_video(
    video_path: Path, timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS
) -> tuple[int, int, float, Optional[float]]:
    """Returns validated video metadata without the codec."""
    width, height, duration, fps, _codec = probe_video_with_codec(video_path, timeout)
    return width, height, duration, fps


def decoder_frame_count(video_path: Path, timeout: float = VIDEO_DECODE_TIMEOUT_SECONDS) -> Optional[int]:
    """Returns the exact decoded frame count, or None if it cannot be determined in time.

    Preferred over a ``duration * fps`` estimate, which can overshoot by one on VFR
    uploads or containers with imprecise metadata; callers fall back to that estimate
    when this returns None.
    """
    result = _run_worker(["count", str(video_path)], timeout)
    if result is None:
        return None
    count = result.get("count")
    if isinstance(count, bool) or not isinstance(count, (int, float)):
        return None
    return int(count) if count > 0 else None
