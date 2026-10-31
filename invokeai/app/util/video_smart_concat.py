"""GOP-aligned planning and bounded bridge encoding for ``video_concat``."""

import math
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeAlias

import imageio_ffmpeg
import numpy as np

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.h264_splice import MIN_ENCODED_FRAMES, H264Layout, SpliceError, read_h264_layout
from invokeai.app.util.video_encoding import CRF, KEYFRAME_INTERVAL_SECONDS

Probe = tuple[int, int, Optional[float], Optional[float]]
_bridge_sample_entry_cache: dict[tuple[tuple[int, int], float], bytes] = {}


@dataclass(frozen=True)
class CopySegment:
    """A source range preserved as compressed H.264 samples."""

    clip: int
    start: int
    end: int


@dataclass(frozen=True)
class BridgeSegment:
    """A single continuously encoded range joining two adjacent source clips."""

    left_clip: int
    right_clip: int
    left_start: int
    right_end: int
    output_frames: int


SmartSegment: TypeAlias = CopySegment | BridgeSegment


@dataclass(frozen=True)
class SmartRenderPlan:
    """A complete, frame-counted plan for one concatenated output timeline."""

    layouts: tuple[H264Layout, ...]
    output_fps: float
    segments: tuple[SmartSegment, ...]
    frame_counts: tuple[int, ...]
    output_frames: int
    copied_frames: int
    encoded_frames: int
    bridge_count: int


def try_build_smart_render_plan(
    paths: list[Path],
    probes: list[Probe],
    canvas: tuple[int, int],
    output_fps: float,
    fps_is_override: bool,
    transition: str,
    transition_frames: int,
    tail_need: int,
    head_need: int,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> tuple[Optional[SmartRenderPlan], str]:
    """Build a conservative all-input plan. Unsupported layouts return a reason for fallback."""
    if is_canceled is not None and is_canceled():
        raise CanceledException
    if len(paths) != len(probes):
        raise ValueError("Smart-render preflight requires one probe result per input path")
    if any((width, height) != canvas for width, height, _, _ in probes):
        return None, "one or more clips need resizing"
    if any(fps is None or not math.isclose(fps, output_fps, rel_tol=1e-6) for _, _, _, fps in probes):
        return None, "one or more clip frame rates do not match the output rate"
    layouts: list[H264Layout] = []
    for index, path in enumerate(paths):
        if is_canceled is not None and is_canceled():
            raise CanceledException
        layout = read_h264_layout(path)
        if is_canceled is not None and is_canceled():
            raise CanceledException
        if layout is None or layout.frame_duration is None or not layout.access_points:
            return None, f"clip {index + 1} is not a verified compatible CFR H.264 MP4"
        layouts.append(layout)

    if is_canceled is not None and is_canceled():
        raise CanceledException
    exact_rates = [layout.time_scale / layout.frame_duration for layout in layouts]
    exact_rate = exact_rates[0]
    if any(not math.isclose(rate, exact_rate, rel_tol=1e-6) for rate in exact_rates[1:]):
        return None, "one or more clips have different exact sample timing"
    if fps_is_override:
        if not math.isclose(exact_rate, output_fps, rel_tol=1e-6):
            return None, "exact sample timing does not match the explicit output frame rate"
        plan_output_fps = output_fps
    else:
        # The isolated decoder probe reports FPS to 0.01 precision. Use that known
        # quantization only to compare metadata; exact sample-table timing remains authoritative.
        probe_fps_tolerance = 0.005
        if (
            any(rate is None or abs(rate - exact_rate) > probe_fps_tolerance for (_, _, _, rate) in probes)
            or abs(output_fps - exact_rate) > probe_fps_tolerance
        ):
            return None, "probe frame rate does not agree with exact MP4 sample timing"
        plan_output_fps = exact_rate
    # Copied samples define the shared sample entry. Any encoded bridge is checked against it.
    target_entry = layouts[0].sample_entry
    for index, (layout, (_, _, duration, _)) in enumerate(zip(layouts, probes, strict=True)):
        if layout.sample_entry != target_entry:
            return None, f"clip {index + 1} is not a verified compatible CFR H.264 MP4"
        if not any(point.sample_index == 0 and point.frame_index == 0 for point in layout.access_points):
            return None, f"clip {index + 1} does not begin at a verified IDR access point"
        expected_duration = layout.sample_count / exact_rate
        if duration is None or abs(duration - expected_duration) > 0.5 / output_fps + 0.01:
            return None, f"clip {index + 1} duration does not match its exact frame timing"

    counts = [layout.sample_count for layout in layouts]
    if tail_need == 0 and head_need == 0:
        transition = "cut"
    if head_need + tail_need and any(
        count < (head_need if i else 0) + (tail_need if i < len(counts) - 1 else 0) for i, count in enumerate(counts)
    ):
        return None, "one or more clips are too short for the requested transition"

    if tail_need == 0 and head_need == 0 and all(count >= MIN_ENCODED_FRAMES for count in counts):
        segments: list[SmartSegment] = [CopySegment(i, 0, count) for i, count in enumerate(counts)]
        copied = sum(counts)
        output_frames = copied
        return (
            SmartRenderPlan(
                tuple(layouts), plan_output_fps, tuple(segments), tuple(counts), output_frames, copied, 0, 0
            ),
            "eligible",
        )

    if transition == "cut" and any(count < MIN_ENCODED_FRAMES for count in counts[1:-1]):
        return None, "a short intermediate clip cannot be covered without overlapping encoded ranges"

    if transition == "cut":
        if len(counts) == 2 and all(count < MIN_ENCODED_FRAMES for count in counts):
            return None, "short clips cannot be covered by one encoded bridge"
        segments: list[SmartSegment] = []
        starts = [0] * len(counts)
        ends = counts.copy()
        bridges: list[BridgeSegment] = []
        if counts[0] < MIN_ENCODED_FRAMES:
            candidates = [frame for _, frame in _safe_points(layouts[1]) if frame > 0]
            right_end = next(
                (
                    frame
                    for frame in candidates
                    if counts[0] + frame >= MIN_ENCODED_FRAMES
                    and (counts[1] - frame == 0 or counts[1] - frame >= MIN_ENCODED_FRAMES)
                ),
                None,
            )
            if right_end is None:
                return None, "the short first clip cannot form a valid encoded bridge"
            bridges.append(BridgeSegment(0, 1, 0, right_end, counts[0] + right_end))
            ends[0] = 0
            starts[1] = right_end
        if counts[-1] < MIN_ENCODED_FRAMES:
            candidates = [frame for _, frame in _safe_points(layouts[-2]) if frame < counts[-2]]
            left_start = next(
                (
                    frame
                    for frame in reversed(candidates)
                    if counts[-2] - frame + counts[-1] >= MIN_ENCODED_FRAMES
                    and (frame - starts[-2] == 0 or frame - starts[-2] >= MIN_ENCODED_FRAMES)
                ),
                None,
            )
            if left_start is None:
                return None, "the short final clip cannot form a valid encoded bridge"
            starts[-1] = counts[-1]
            ends[-2] = left_start
            bridges.append(
                BridgeSegment(
                    len(counts) - 2, len(counts) - 1, left_start, counts[-1], counts[-1] + counts[-2] - left_start
                )
            )
        if bridges:
            expected_entry = _bridge_sample_entry(canvas, plan_output_fps, is_canceled)
            if expected_entry is None or target_entry != expected_entry:
                return None, "encoded bridges require the current bridge encoder H.264 sample entry"
        bridges_by_left = {segment.left_clip: segment for segment in bridges}
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if start < end:
                if end - start < MIN_ENCODED_FRAMES:
                    return None, f"copied clip {i + 1} interval is too short"
                segments.append(CopySegment(i, start, end))
            bridge = bridges_by_left.get(i)
            if bridge is not None:
                segments.append(bridge)
        copied = sum(segment.end - segment.start for segment in segments if isinstance(segment, CopySegment))
        encoded = sum(segment.output_frames for segment in bridges)
        if copied + encoded != sum(counts):
            return None, "the plan does not preserve an exact output frame partition"
        return (
            SmartRenderPlan(
                tuple(layouts),
                plan_output_fps,
                tuple(segments),
                tuple(counts),
                sum(counts),
                copied,
                encoded,
                len(bridges),
            ),
            "eligible",
        )

    expected_entry = _bridge_sample_entry(canvas, plan_output_fps, is_canceled)
    if expected_entry is None or target_entry != expected_entry:
        return None, "encoded bridges require the current bridge encoder H.264 sample entry"

    boundaries: list[tuple[int, int, int]] = []
    encoded_counts: list[int] = []
    for i in range(len(counts) - 1):
        left = _safe_points(layouts[i])
        right = _safe_points(layouts[i + 1])
        left_limit = counts[i] - tail_need
        left_candidates = [(sample, frame) for sample, frame in left if frame <= left_limit]
        right_candidates = [(sample, frame) for sample, frame in right if frame >= head_need]
        if transition == "cut" and i + 1 == len(counts) - 1 and counts[-1] < MIN_ENCODED_FRAMES:
            # A short final clip cannot be copied as a separate part: encode it into the
            # preceding tail, extending that tail back to a safe IDR as needed.
            right_candidates = [(counts[i + 1], counts[i + 1])]
        if not left_candidates:
            left_candidates = [(0, 0)]
        if not right_candidates or right_candidates[-1][1] != counts[i + 1]:
            # The source edge is always a safe endpoint: no packet follows it to depend on.
            right_candidates.append((counts[i + 1], counts[i + 1]))
        left_candidates = sorted(set(left_candidates), key=lambda pair: pair[1], reverse=True)
        right_candidates = sorted(set(right_candidates), key=lambda pair: pair[1])
        selected: Optional[tuple[int, int, int]] = None
        for left_sample, left_frame in left_candidates:
            for right_sample, right_frame in right_candidates:
                encoded = counts[i] - left_frame + right_frame
                if transition == "crossfade":
                    encoded -= transition_frames
                if encoded >= MIN_ENCODED_FRAMES:
                    selected = (left_sample, right_sample, encoded)
                    break
            if selected is not None:
                break
        if selected is None:
            return None, f"transition {i + 1} cannot form a valid encoded bridge"
        boundaries.append(selected)
        encoded_counts.append(selected[2])

    segments = []
    first_start = boundaries[0][0]
    if first_start:
        segments.append(CopySegment(0, 0, first_start))
    for i, (left_start, right_end, encoded) in enumerate(boundaries):
        segments.append(BridgeSegment(i, i + 1, left_start, right_end, encoded))
        if i + 1 < len(boundaries):
            next_start = boundaries[i + 1][0]
            if right_end > next_start:
                return None, f"the bridge ranges overlap around clip {i + 2}"
            if right_end < next_start:
                segments.append(CopySegment(i + 1, right_end, next_start))
    last_end = boundaries[-1][1]
    if last_end < counts[-1]:
        segments.append(CopySegment(len(counts) - 1, last_end, counts[-1]))

    copy_segments = [segment for segment in segments if isinstance(segment, CopySegment)]
    if any(segment.end - segment.start < MIN_ENCODED_FRAMES for segment in copy_segments):
        return None, "a copied interval is too short to preserve a continuous packet timeline"
    copied = sum(segment.end - segment.start for segment in copy_segments)
    encoded = sum(encoded_counts)
    output_frames = sum(counts) - (transition_frames * (len(counts) - 1) if transition == "crossfade" else 0)
    if copied == 0 or copied + encoded != output_frames:
        return None, "the plan does not preserve an exact output frame partition"
    return (
        SmartRenderPlan(
            tuple(layouts),
            plan_output_fps,
            tuple(segments),
            tuple(counts),
            output_frames,
            copied,
            encoded,
            len(boundaries),
        ),
        "eligible",
    )


def _safe_points(layout: H264Layout) -> list[tuple[int, int]]:
    """Return only IDRs whose packet and presentation indices identify the same frame."""
    return [
        (point.sample_index, point.frame_index)
        for point in layout.access_points
        if point.sample_index == point.frame_index
    ]


def _bridge_sample_entry(
    canvas: tuple[int, int], fps: float, is_canceled: Optional[Callable[[], bool]] = None
) -> Optional[bytes]:
    """Probe the direct FFmpeg encoder used for bridge parts."""
    if is_canceled is not None and is_canceled():
        raise CanceledException
    key = (canvas, fps)
    cached = _bridge_sample_entry_cache.get(key)
    if cached is not None:
        if is_canceled is not None and is_canceled():
            raise CanceledException
        return cached
    with tempfile.TemporaryDirectory(prefix="invokeai_bridge_probe_") as work:
        path = Path(work) / "probe.mp4"
        try:
            encode_rgb_frames(
                (np.zeros((canvas[1], canvas[0], 3), dtype=np.uint8) for _ in range(MIN_ENCODED_FRAMES)),
                path,
                canvas,
                fps,
                MIN_ENCODED_FRAMES,
                is_canceled=is_canceled,
            )
        except SpliceError:
            return None
        layout = read_h264_layout(path)
    if is_canceled is not None and is_canceled():
        raise CanceledException
    if layout is None:
        return None
    _bridge_sample_entry_cache[key] = layout.sample_entry
    return layout.sample_entry


def encode_rgb_frames(
    frames: Iterable[np.ndarray],
    output_path: Path,
    canvas: tuple[int, int],
    fps: float,
    expected_frames: int,
    is_canceled: Optional[Callable[[], bool]] = None,
    abort_event: Optional[threading.Event] = None,
    timeout: float = 600,
) -> int:
    """Stream RGB frames to bounded, cancellable bundled FFmpeg H.264 encoding."""
    width, height = canvas
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        format(fps, ".12g"),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(CRF),
        "-g",
        str(max(1, round(fps * KEYFRAME_INTERVAL_SECONDS))),
        "-frames:v",
        str(expected_frames),
        str(output_path),
    ]
    if expected_frames < MIN_ENCODED_FRAMES:
        raise ValueError(f"A smart-render bridge must contain at least {MIN_ENCODED_FRAMES} frames")
    if is_canceled is not None and is_canceled():
        raise CanceledException
    count = 0
    failures: list[BaseException] = []
    stopped = abort_event or threading.Event()
    stderr_limit = 64 * 1024
    stderr_tail = bytearray()
    stderr_lock = threading.Lock()
    process: Optional[subprocess.Popen[bytes]] = None
    feeder: Optional[threading.Thread] = None
    stderr_reader: Optional[threading.Thread] = None
    canceled = False
    timed_out = False

    try:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as e:
            raise SpliceError(f"could not start ffmpeg to encode a transition bridge: {e}") from e

        def read_stderr() -> None:
            assert process is not None and process.stderr is not None
            while chunk := process.stderr.read(8192):
                with stderr_lock:
                    stderr_tail.extend(chunk)
                    if len(stderr_tail) > stderr_limit:
                        del stderr_tail[: len(stderr_tail) - stderr_limit]

        def feed() -> None:
            nonlocal count
            try:
                assert process is not None and process.stdin is not None
                for frame in frames:
                    if stopped.is_set() or (is_canceled is not None and is_canceled()):
                        raise RuntimeError("bridge frame input stopped")
                    if count >= expected_frames:
                        raise SpliceError(f"bridge input exceeded its planned {expected_frames} frames")
                    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                        raise ValueError(
                            f"Expected uint8 RGB frames at {width}x{height}, got {frame.shape} {frame.dtype}"
                        )
                    data = memoryview(np.ascontiguousarray(frame)).cast("B")
                    while data:
                        written = process.stdin.write(data)
                        if written is None or written == 0:
                            raise BrokenPipeError("ffmpeg stopped reading bridge frames")
                        data = data[written:]
                    count += 1
            except BaseException as e:
                failures.append(e)
            finally:
                try:
                    if process is not None and process.stdin is not None:
                        process.stdin.close()
                except OSError:
                    pass

        stderr_reader = threading.Thread(target=read_stderr, name="invokeai-video-bridge-stderr", daemon=True)
        feeder = threading.Thread(target=feed, name="invokeai-video-bridge-feed", daemon=True)
        try:
            stderr_reader.start()
            feeder.start()
        except RuntimeError as e:
            raise SpliceError(f"could not start bridge worker thread: {e}") from e

        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if is_canceled is not None and is_canceled():
                canceled = True
                stopped.set()
                break
            if time.monotonic() >= deadline:
                timed_out = True
                stopped.set()
                break
            time.sleep(0.05)
        if canceled or timed_out:
            process.kill()
        elif process.returncode is not None and process.returncode != 0:
            stopped.set()
        process.wait()
        feeder.join(timeout=35)
        if feeder.is_alive():
            stopped.set()
            process.kill()
            process.wait()
            feeder.join(timeout=5)
        stderr_reader.join(timeout=5)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        # Any exception after Popen must stop the child and both workers before it escapes.
        stopped.set()
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
            except BaseException:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except BaseException:
                    pass
                process.wait()
            except BaseException:
                pass
        for worker in (feeder, stderr_reader):
            if worker is not None and worker.is_alive():
                worker.join(timeout=5)
        if process is not None:
            for stream in (process.stdin, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    diagnostic = bytes(stderr_tail).decode("utf-8", errors="replace").strip()
    if feeder is not None and feeder.is_alive():
        output_path.unlink(missing_ok=True)
        raise SpliceError("the bridge encoder did not stop after its input was closed")
    if stderr_reader is not None and stderr_reader.is_alive():
        output_path.unlink(missing_ok=True)
        raise SpliceError("the bridge encoder diagnostic reader did not stop")
    if canceled or (is_canceled is not None and is_canceled()):
        output_path.unlink(missing_ok=True)
        raise CanceledException
    if timed_out:
        output_path.unlink(missing_ok=True)
        raise SpliceError("ffmpeg timed out encoding a transition bridge")
    assert process is not None
    if process.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise SpliceError(f"ffmpeg failed encoding a transition bridge: {diagnostic[-500:]}")
    if failures:
        failure = failures[0]
        if isinstance(failure, CanceledException):
            output_path.unlink(missing_ok=True)
            raise failure
        if isinstance(failure, BrokenPipeError):
            output_path.unlink(missing_ok=True)
            raise SpliceError(f"ffmpeg closed the transition bridge input early: {diagnostic[-500:]}") from failure
        if isinstance(failure, ValueError):
            output_path.unlink(missing_ok=True)
            raise SpliceError(f"could not decode or prepare transition frames: {failure}") from failure
        if isinstance(failure, SpliceError):
            output_path.unlink(missing_ok=True)
            raise failure
        output_path.unlink(missing_ok=True)
        raise failure
    if count != expected_frames:
        output_path.unlink(missing_ok=True)
        raise SpliceError(f"bridge input contained {count} frames; expected {expected_frames}")
    return count
