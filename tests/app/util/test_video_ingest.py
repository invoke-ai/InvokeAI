"""Tests for upload-time media normalization (video_ingest).

Fixtures are generated with the same bundled ffmpeg the module invokes, using codecs
that binary can *encode* (h264/mpeg4/aac/pcm) — HEVC and ProRes can only be decoded,
so the "foreign codec" transcode branch is exercised with mpeg4, which takes the same
code path.
"""

import subprocess
from pathlib import Path

import pytest

from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_ingest import (
    MediaProbe,
    VideoIngestError,
    ingest_media_to_mp4,
    plan_ingest,
    probe_media_streams,
)
from invokeai.app.util.video_thumbnails import probe_video_with_codec

GB = 1 << 30


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_media(path: Path, *args: str) -> Path:
    subprocess.run([_ffmpeg_exe(), "-y", "-loglevel", "error", *args, str(path)], check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("ingest_fixtures")


@pytest.fixture(scope="session")
def h264_mov(fixture_dir: Path) -> Path:
    """H.264 + AAC in a QuickTime container — the iPhone 'Most Compatible' shape."""
    return _make_media(
        fixture_dir / "clip.mov",
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=8:d=1"),
        *("-f", "lavfi", "-i", "sine=frequency=440:d=1"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"),
    )


@pytest.fixture(scope="session")
def mpeg4_avi(fixture_dir: Path) -> Path:
    """A decodable-but-foreign video codec, standing in for HEVC/ProRes."""
    return _make_media(
        fixture_dir / "clip.avi",
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=8:d=1"),
        *("-c:v", "mpeg4"),
    )


@pytest.fixture(scope="session")
def wav_audio(fixture_dir: Path) -> Path:
    return _make_media(
        fixture_dir / "tone.wav",
        *("-f", "lavfi", "-i", "anoisesrc=a=0.3:d=1"),
        *("-c:a", "pcm_s16le"),
    )


@pytest.fixture(scope="session")
def m4a_audio(fixture_dir: Path) -> Path:
    return _make_media(
        fixture_dir / "tone.m4a",
        *("-f", "lavfi", "-i", "sine=frequency=550:d=1"),
        *("-c:a", "aac"),
    )


def test_probe_reports_video_and_audio_codecs(h264_mov: Path, wav_audio: Path) -> None:
    probe = probe_media_streams(h264_mov)
    assert probe.video_codec == "h264"
    assert probe.audio_codec == "aac"

    probe = probe_media_streams(wav_audio)
    assert probe.video_codec is None
    assert probe.audio_codec is not None


def test_probe_rejects_unreadable_bytes(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mov"
    garbage.write_bytes(b"this is not media" * 64)
    with pytest.raises(VideoIngestError):
        probe_media_streams(garbage)


def test_plan_covers_each_stream_layout() -> None:
    assert plan_ingest(MediaProbe(video_codec="h264", audio_codec=None)) == "remux"
    assert plan_ingest(MediaProbe(video_codec="hevc", audio_codec="aac")) == "transcode"
    assert plan_ingest(MediaProbe(video_codec=None, audio_codec="mp3")) == "audio_wrap"
    with pytest.raises(VideoIngestError):
        plan_ingest(MediaProbe(video_codec=None, audio_codec=None))


def test_h264_foreign_container_is_remuxed(h264_mov: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out.mp4"
    assert ingest_media_to_mp4(h264_mov, dst, max_output_bytes=GB) == "remux"
    width, height, _duration, _fps, codec = probe_video_with_codec(dst)
    assert (width, height, codec) == (64, 48, "h264")
    pcm = extract_audio_pcm(dst)
    assert pcm is not None, "the AAC track must survive the remux"


def test_foreign_codec_is_transcoded_to_h264(mpeg4_avi: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out.mp4"
    assert ingest_media_to_mp4(mpeg4_avi, dst, max_output_bytes=GB) == "transcode"
    width, height, _duration, _fps, codec = probe_video_with_codec(dst)
    assert (width, height, codec) == (64, 48, "h264")


def test_odd_dimensions_are_evened_for_yuv420p(fixture_dir: Path, tmp_path: Path) -> None:
    src = _make_media(
        fixture_dir / "odd.avi",
        *("-f", "lavfi", "-i", "testsrc2=s=63x47:r=8:d=1"),
        *("-c:v", "mpeg4"),
    )
    dst = tmp_path / "out.mp4"
    assert ingest_media_to_mp4(src, dst, max_output_bytes=GB) == "transcode"
    width, height, _duration, _fps, codec = probe_video_with_codec(dst)
    assert (width, height, codec) == (62, 46, "h264")


@pytest.mark.parametrize("fixture_name", ["wav_audio", "m4a_audio"])
def test_audio_only_uploads_are_wrapped_into_waveform_videos(
    fixture_name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    src: Path = request.getfixturevalue(fixture_name)
    dst = tmp_path / "out.mp4"
    assert ingest_media_to_mp4(src, dst, max_output_bytes=GB) == "audio_wrap"
    width, height, duration, fps, codec = probe_video_with_codec(dst)
    assert (width, height, codec) == (640, 360, "h264")
    assert fps == 24.0
    # The video track must span the audio (AAC priming may pad slightly past it).
    assert 0.9 <= duration <= 1.5
    pcm = extract_audio_pcm(dst)
    assert pcm is not None


def test_album_art_does_not_make_an_audio_file_a_video(fixture_dir: Path, tmp_path: Path) -> None:
    """An m4a with embedded cover art carries a real (attached-pic) video stream; it must
    still be classified audio-only and wrapped, not 'transcoded' into a still image."""
    cover = _make_media(
        fixture_dir / "cover_src.mp4",
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=1:d=1"),
        *("-frames:v", "1", "-c:v", "mjpeg"),
    )
    src = fixture_dir / "with_art.m4a"
    subprocess.run(
        [
            _ffmpeg_exe(),
            *("-y", "-loglevel", "error"),
            *("-f", "lavfi", "-i", "sine=frequency=330:d=1"),
            *("-i", str(cover)),
            *("-map", "0:a", "-map", "1:v"),
            *("-c:a", "aac", "-c:v", "copy"),
            *("-disposition:v:0", "attached_pic"),
            # The ipod/m4a muxer refuses mjpeg streams; the plain mp4 muxer takes them,
            # and readers sniff content rather than trusting the .m4a name.
            *("-f", "mp4"),
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    probe = probe_media_streams(src)
    assert probe.video_codec is None, "attached cover art must not count as a video stream"

    dst = tmp_path / "out.mp4"
    assert ingest_media_to_mp4(src, dst, max_output_bytes=GB) == "audio_wrap"


def test_unreadable_input_raises_user_safe_error(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp3"
    garbage.write_bytes(b"\x00" * 1024)
    with pytest.raises(VideoIngestError):
        ingest_media_to_mp4(garbage, tmp_path / "out.mp4", max_output_bytes=GB)
    assert not (tmp_path / "out.mp4").exists() or (tmp_path / "out.mp4").stat().st_size == 0


def test_output_size_cap_is_enforced(h264_mov: Path, tmp_path: Path) -> None:
    with pytest.raises(VideoIngestError, match="maximum upload size"):
        ingest_media_to_mp4(h264_mov, tmp_path / "out.mp4", max_output_bytes=64)
