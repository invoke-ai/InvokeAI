"""Tests for the LlavaOnevisionPipeline class."""

import threading
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from invokeai.backend.llava_onevision_pipeline import LlavaOnevisionPipeline


def _make_mock_processor() -> MagicMock:
    """Create a mock LLaVA processor whose tokenizer reports a fixed token count."""
    processor = MagicMock()
    processor.apply_chat_template.return_value = "<formatted prompt>"

    processor_output = MagicMock()
    processor_output.to.return_value = processor_output
    processor.return_value = processor_output

    processor.tokenizer = MagicMock()
    processor.tokenizer.encode.return_value = [10, 11, 12]
    return processor


def _make_image() -> Image.Image:
    return Image.new("RGB", (8, 8))


class FakeStreamer:
    """Stand-in for TextIteratorStreamer — yields a fixed sequence of text chunks."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


def _patch_streamer(chunks: list[str] | None = None):
    """Patch TextIteratorStreamer in the pipeline module to return a FakeStreamer."""
    chunks = chunks if chunks is not None else ["A cat ", "on a ", "sofa"]
    return patch(
        "invokeai.backend.llava_onevision_pipeline.TextIteratorStreamer",
        return_value=FakeStreamer(chunks),
    )


def test_pipeline_returns_joined_streamed_chunks():
    """Pipeline should return the concatenated, stripped streamer output.

    This exercises the streamer-based output parsing (skip_prompt=True) that replaced
    the earlier split("assistant\\n") hack — the returned text is exactly the streamed
    chunks with no prompt echo to strip.
    """
    processor = _make_mock_processor()
    model = MagicMock()
    pipeline = LlavaOnevisionPipeline(model, processor)

    with _patch_streamer(["  a scenic ", "mountain view  "]):
        result = pipeline.run(
            prompt="describe",
            images=[_make_image()],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert result == "a scenic mountain view"


def test_pipeline_invokes_progress_callback():
    """Pipeline should report generation progress via progress_callback.

    Emissions are throttled, so the callback fires at least once, always reports the
    configured total, and its final call reflects the true accumulated token count.
    """
    processor = _make_mock_processor()
    model = MagicMock()
    pipeline = LlavaOnevisionPipeline(model, processor)
    calls: list[tuple[int, int]] = []

    with _patch_streamer(["a ", "b ", "c"]):
        pipeline.run(
            prompt="describe",
            images=[_make_image()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_new_tokens=50,
            progress_callback=lambda current, total: calls.append((current, total)),
        )

    assert len(calls) >= 1
    assert all(total == 50 for _, total in calls)
    assert calls[-1] == (3, 50)


def test_pipeline_rejects_too_many_images():
    """Pipeline should reject more than 3 images."""
    processor = _make_mock_processor()
    model = MagicMock()
    pipeline = LlavaOnevisionPipeline(model, processor)

    try:
        pipeline.run(
            prompt="describe",
            images=[_make_image() for _ in range(4)],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        raise AssertionError("expected ValueError for >3 images")
    except ValueError as e:
        assert "images" in str(e)


def test_pipeline_reraises_generation_error_without_hanging():
    """If generate() raises in the worker thread, run() must re-raise it promptly rather
    than deadlock on the streamer.

    Uses the real TextIteratorStreamer (not FakeStreamer) so the test exercises the
    streamer.end()-on-exception path that unblocks the consumer loop.
    """
    processor = _make_mock_processor()
    model = MagicMock()
    model.generate.side_effect = RuntimeError("CUDA out of memory")
    pipeline = LlavaOnevisionPipeline(model, processor)

    result: list[BaseException] = []

    def _run() -> None:
        try:
            pipeline.run(
                prompt="describe",
                images=[_make_image()],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        except BaseException as e:  # noqa: BLE001 - capture whatever run() raises
            result.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=10.0)

    assert not thread.is_alive(), "pipeline.run() deadlocked when generate() raised"
    assert len(result) == 1
    assert isinstance(result[0], RuntimeError)
    assert "CUDA out of memory" in str(result[0])
