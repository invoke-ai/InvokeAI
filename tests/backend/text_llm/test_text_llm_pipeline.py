"""Tests for the TextLLMPipeline class."""

import threading
from unittest.mock import MagicMock, patch

import torch

from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT, TextLLMPipeline


def _make_mock_tokenizer(has_chat_template: bool = True) -> MagicMock:
    """Create a mock tokenizer with configurable chat template support."""
    tokenizer = MagicMock()
    if has_chat_template:
        tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        tokenizer.apply_chat_template.return_value = "<|system|>You are helpful<|user|>hello<|assistant|>"
    else:
        tokenizer.chat_template = None

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer_output = MagicMock()
    tokenizer_output.__getitem__ = lambda self, key: {"input_ids": input_ids}[key]
    tokenizer_output.to.return_value = tokenizer_output
    tokenizer.return_value = tokenizer_output

    # Token-counting for progress: pretend each accumulated string is N tokens long.
    tokenizer.encode.return_value = [10, 11, 12]
    return tokenizer


def _make_mock_model() -> MagicMock:
    return MagicMock()


class FakeStreamer:
    """Stand-in for TextIteratorStreamer — yields a fixed sequence of text chunks."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


def _patch_streamer(chunks: list[str] | None = None):
    """Patch TextIteratorStreamer in the pipeline module to return a FakeStreamer."""
    chunks = chunks if chunks is not None else ["A detailed ", "landscape ", "with mountains"]
    return patch(
        "invokeai.backend.text_llm_pipeline.TextIteratorStreamer",
        return_value=FakeStreamer(chunks),
    )


def test_pipeline_uses_chat_template_when_available():
    """Pipeline should use apply_chat_template when the tokenizer supports it."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a cat", device=torch.device("cpu"), dtype=torch.float32)

    tokenizer.apply_chat_template.assert_called_once()
    call_args = tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" and m["content"] == "a cat" for m in messages)


def test_pipeline_fallback_without_chat_template():
    """Pipeline should use fallback formatting when no chat template exists."""
    tokenizer = _make_mock_tokenizer(has_chat_template=False)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a cat", system_prompt="Be helpful", device=torch.device("cpu"), dtype=torch.float32)

    tokenizer.apply_chat_template.assert_not_called()
    call_args = tokenizer.call_args[0][0]
    assert "Be helpful" in call_args
    assert "a cat" in call_args
    assert "Assistant:" in call_args


def test_pipeline_no_system_prompt():
    """Pipeline should work without a system prompt."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a dog", system_prompt="", device=torch.device("cpu"), dtype=torch.float32)

    call_args = tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    assert not any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" and m["content"] == "a dog" for m in messages)


def test_pipeline_passes_generation_params():
    """Pipeline should pass max_new_tokens and sampling params to model.generate, plus a streamer."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="test", max_new_tokens=100, device=torch.device("cpu"), dtype=torch.float32)

    generate_kwargs = model.generate.call_args[1]
    assert generate_kwargs["max_new_tokens"] == 100
    assert generate_kwargs["do_sample"] is True
    assert generate_kwargs["temperature"] == 0.7
    assert generate_kwargs["top_p"] == 0.9
    assert "streamer" in generate_kwargs


def test_pipeline_returns_joined_streamed_chunks():
    """Pipeline should return the concatenated, stripped streamer output."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    with _patch_streamer(["  hello ", "world  "]):
        result = pipeline.run(prompt="test", device=torch.device("cpu"), dtype=torch.float32)

    assert result == "hello world"


def test_pipeline_invokes_progress_callback():
    """Pipeline should report generation progress via progress_callback.

    Emissions are throttled, so the callback is not guaranteed to fire once per chunk,
    but it must fire at least once, always report the configured total, and its final
    call must reflect the true accumulated token count.
    """
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)
    calls: list[tuple[int, int]] = []

    with _patch_streamer(["a ", "b ", "c"]):
        pipeline.run(
            prompt="test",
            max_new_tokens=50,
            device=torch.device("cpu"),
            dtype=torch.float32,
            progress_callback=lambda current, total: calls.append((current, total)),
        )

    assert len(calls) >= 1
    assert all(total == 50 for _, total in calls)
    # The mock tokenizer reports 3 tokens for the accumulated text; the final emission
    # must reflect that (clamped to max_new_tokens).
    assert calls[-1] == (3, 50)


def test_pipeline_reraises_generation_error_without_hanging():
    """If model.generate() raises in the worker thread, run() must re-raise it promptly
    rather than deadlock on the streamer.

    Uses the real TextIteratorStreamer (not FakeStreamer) so the test exercises the
    streamer.end()-on-exception path that unblocks the consumer loop. The generous
    STREAM_TIMEOUT means a regression here would hang for two minutes, so the test is
    wrapped in a hard timeout to fail fast instead.
    """
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    model.generate.side_effect = RuntimeError("CUDA out of memory")
    pipeline = TextLLMPipeline(model, tokenizer)

    result: list[BaseException] = []

    def _run() -> None:
        try:
            pipeline.run(prompt="test", device=torch.device("cpu"), dtype=torch.float32)
        except BaseException as e:  # noqa: BLE001 - capture whatever run() raises
            result.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=10.0)

    assert not thread.is_alive(), "pipeline.run() deadlocked when generate() raised"
    assert len(result) == 1
    assert isinstance(result[0], RuntimeError)
    assert "CUDA out of memory" in str(result[0])


def test_default_system_prompt_content():
    """The default system prompt should mention image generation."""
    assert "image generation" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "prompt" in DEFAULT_SYSTEM_PROMPT.lower()
