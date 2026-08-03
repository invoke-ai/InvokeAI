"""Tests for the TextLLMPipeline class."""

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend import text_llm_pipeline
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT, TextLLMPipeline, _SeededMultinomialMode


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


def test_seeded_multinomial_is_repeatable_despite_concurrent_global_rng_use():
    """Unrelated RNG use must not alter sampling for a controlled seed."""
    probabilities = torch.ones(100)
    global_rng_state = torch.random.get_rng_state()

    with _SeededMultinomialMode(seed=42):
        expected = torch.multinomial(probabilities, num_samples=10, replacement=True)

    assert torch.equal(torch.random.get_rng_state(), global_rng_state)

    interference_done = threading.Event()

    def _interfere() -> None:
        torch.multinomial(probabilities, num_samples=10, replacement=True)
        interference_done.set()

    with _SeededMultinomialMode(seed=42):
        thread = threading.Thread(target=_interfere)
        thread.start()
        assert interference_done.wait(timeout=1)
        actual = torch.multinomial(probabilities, num_samples=10, replacement=True)
        thread.join()

    assert torch.equal(actual, expected)


def test_seeded_multinomial_contexts_are_isolated_when_interleaved_on_cpu():
    """Concurrent seeded contexts must retain independent RNG sequences."""
    probabilities = torch.arange(1, 101, dtype=torch.float32)
    seeds = (42, 1234)
    draw_count = 100

    expected: dict[int, list[int]] = {}
    for seed in seeds:
        with _SeededMultinomialMode(seed=seed):
            expected[seed] = [torch.multinomial(probabilities, num_samples=1).item() for _ in range(draw_count)]

    barrier = threading.Barrier(len(seeds))
    actual: dict[int, list[int]] = {}

    def _sample(seed: int) -> None:
        samples: list[int] = []
        with _SeededMultinomialMode(seed=seed):
            for _ in range(draw_count):
                barrier.wait(timeout=5)
                samples.append(torch.multinomial(probabilities, num_samples=1).item())
        actual[seed] = samples

    threads = [threading.Thread(target=_sample, args=(seed,), daemon=True) for seed in seeds]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert actual == expected


def test_stalled_generation_does_not_block_later_generation(monkeypatch: pytest.MonkeyPatch):
    """A timed-out worker must not own shared state needed by later runs."""
    monkeypatch.setattr(text_llm_pipeline, "STREAM_TIMEOUT", 0.05)
    stalled_model = MagicMock()
    release_stalled_model = threading.Event()
    stalled_model.generate.side_effect = lambda **kwargs: release_stalled_model.wait()

    with pytest.raises(RuntimeError, match="Text generation stalled"):
        TextLLMPipeline(stalled_model, _make_mock_tokenizer()).run("test", device=torch.device("cpu"))

    healthy_model = MagicMock()

    def _generate(**kwargs):
        streamer = kwargs["streamer"]
        streamer.put(torch.tensor([[1, 2, 3, 4, 5]]))
        streamer.put(torch.tensor([6]))
        streamer.end()

    healthy_model.generate.side_effect = _generate
    healthy_tokenizer = _make_mock_tokenizer()
    healthy_tokenizer.decode.return_value = "ok"
    try:
        TextLLMPipeline(healthy_model, healthy_tokenizer).run("test", device=torch.device("cpu"))
    finally:
        release_stalled_model.set()

    healthy_model.generate.assert_called_once()


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
