"""Shared fixtures and helpers of the video-concat splice tests: real FFmpeg encodes textured clips and decodes the joins."""

import hashlib
import struct
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import imageio.v2 as iio2
import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation
from invokeai.app.util.h264_splice import read_h264_layout
from invokeai.app.util.mp4_metadata import find_box, write_mp4_tags
from invokeai.app.util.video_encoding import (
    make_mp4_writer,
    write_stereo_wav,
)

FPS = 16.0


WIDTH, HEIGHT = 128, 96


RATE = 32000


def _frames(n: int, seed: int) -> list[np.ndarray]:
    # Textured, moving content: flat frames encode near-losslessly and would hide a re-encode.
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, (HEIGHT + n, WIDTH + n, 3), dtype=np.uint8)
    return [texture[i : i + HEIGHT, i : i + WIDTH].copy() for i in range(n)]


def _tone(freq: float, n_frames: int) -> np.ndarray:
    t = np.arange(round(n_frames / FPS * RATE)) / RATE
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono])


def _encode(path: Path, frames: list[np.ndarray], tone: float | None = None, legacy: bool = False) -> Path:
    audio_path = None
    if tone is not None:
        audio_path = path.with_suffix(".wav")
        write_stereo_wav(audio_path, _tone(tone, len(frames)), RATE)
    if legacy:
        # A file from before the current encoder settings (CRF 25), with keyframes as frequent as
        # ours so a splice point exists and only the decoder configuration rules the copy out.
        writer = iio2.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            output_params=["-crf", "25", "-g", "32"],
        )
    else:
        writer = make_mp4_writer(path, FPS, audio_path=audio_path)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def _encode_open_gop(path: Path, frames: list[np.ndarray]) -> Path:
    writer = iio2.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        quality=None,
        output_params=["-crf", "18", "-g", "32", "-x264-params", "open-gop=1"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def _decode(path: Path) -> list[np.ndarray]:
    reader = iio2.get_reader(str(path), format="FFMPEG")
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def _movie_duration(path: Path) -> float:
    """The container's own duration (``mvhd``), at its full precision."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        moov = find_box(fh, 0, fh.tell(), b"moov")
        assert moov is not None
        mvhd = find_box(fh, *moov, b"mvhd")
        assert mvhd is not None
        fh.seek(mvhd[0])
        wide = fh.read(1)[0] == 1
        fh.seek(mvhd[0] + (20 if wide else 12))
        timescale, duration = struct.unpack(">IQ" if wide else ">II", fh.read(12 if wide else 8))
    return duration / timescale


def _dominant_freq(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm[0]))
    spectrum[0] = 0.0
    return float(np.fft.rfftfreq(pcm.shape[1], 1.0 / rate)[int(np.argmax(spectrum))])


def _sample_payload_hashes(path: Path, start: int, end: int) -> list[str]:
    """Hash coded picture NALs, ignoring legal parameter-set and framing changes."""
    from invokeai.app.util import h264_splice
    from invokeai.app.util.mp4_metadata import find_box, iter_boxes

    with path.open("rb") as stream:
        stream.seek(0, 2)
        moov = find_box(stream, 0, stream.tell(), b"moov")
        assert moov is not None
        stbl = None
        for box_type, trak_start, trak_end in iter_boxes(stream, *moov):
            if box_type != b"trak":
                continue
            mdia = find_box(stream, trak_start, trak_end, b"mdia")
            if mdia is None:
                continue
            hdlr = find_box(stream, *mdia, b"hdlr")
            if hdlr is None:
                continue
            stream.seek(hdlr[0] + 8)
            if stream.read(4) != b"vide":
                continue
            minf = find_box(stream, *mdia, b"minf")
            stbl = find_box(stream, *minf, b"stbl") if minf is not None else None
            break
        assert stbl is not None
        stsz = find_box(stream, *stbl, b"stsz")
        assert stsz is not None
        stream.seek(stsz[0] + 8)
        sample_count = int.from_bytes(stream.read(4), "big")
        stream.seek(0, 2)
        file_size = stream.tell()
        sizes = h264_splice._sample_sizes(stream, stsz, sample_count, file_size)
        assert sizes is not None
        locations = h264_splice._sample_locations(stream, stbl, sizes, tuple(range(start, end)), file_size)
        assert locations is not None
        layout = read_h264_layout(path)
        assert layout is not None
        length_size = (layout.decoder_config[4] & 0x03) + 1
        hashes = []
        for offset, size in locations:
            stream.seek(offset)
            sample = stream.read(size)
            digest = hashlib.sha256()
            position = 0
            has_picture = False
            while position + length_size < len(sample):
                nal_size = int.from_bytes(sample[position : position + length_size], "big")
                position += length_size
                nal = sample[position : position + nal_size]
                assert len(nal) == nal_size
                if nal and (nal[0] & 0x1F) in range(1, 6):
                    digest.update(nal)
                    has_picture = True
                position += nal_size
            assert position == len(sample) and has_picture
            hashes.append(digest.hexdigest())
        return hashes


def _concat(tmp_path: Path, inputs: list[Path], out_name: str, **fields: object) -> tuple[Path, MagicMock]:
    """Run the node on real files; returns the saved MP4 and the mocked context."""
    by_name = {p.stem: p for p in inputs}
    saved = tmp_path / f"{out_name}.mp4"

    def _save(source_path: Path, **_: object) -> MagicMock:
        # The stream-copy remux save() runs to embed the record, so chained joins read what
        # production stores.
        write_mp4_tags(source_path, saved, {"invokeai_metadata": "{}"})
        return MagicMock()

    context = MagicMock()
    context.videos.get_path.side_effect = lambda name: by_name[name]
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save
    node = VideoConcatInvocation(videos=[VideoField(video_name=p.stem) for p in inputs], **fields)  # type: ignore[arg-type]
    with patch("invokeai.app.invocations.video_concat.VideoOutput.build", return_value=MagicMock()):
        node.invoke(context)
    return saved, context


def _logged(context: MagicMock) -> str:
    return "\n".join(str(call.args[0]) for call in context.logger.info.call_args_list)


def _last_usable_keyframe(path: Path, tail: int) -> int:
    layout = read_h264_layout(path)
    assert layout is not None
    return max(k for k in layout.sync_samples if 0 < k <= layout.sample_count - max(tail, 1))


def _remux(src: Path, dst: Path, *args: str) -> Path:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args, "-i", str(src), "-c", "copy", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


def _encode_fractional_rate(
    path: Path, seed: int, frame_count: int = 40, rate: tuple[int, int] = (24000, 1001)
) -> Path:
    frames = np.stack(_frames(frame_count, seed))
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            f"{rate[0]}/{rate[1]}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-g",
            "48",
            "-video_track_timescale",
            str(rate[0]),
            "-frames:v",
            str(len(frames)),
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
        capture_output=True,
    )
    layout = read_h264_layout(path)
    assert layout is not None
    assert layout.time_scale / layout.frame_duration == pytest.approx(rate[0] / rate[1])
    return path
