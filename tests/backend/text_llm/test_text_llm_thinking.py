"""Regression tests for reasoning ("thinking") models in TextLLMPipeline.

Reasoning models such as Qwen3 emit a chain of thought before their answer unless the chat
template is rendered with ``enable_thinking=False``. Prompt expansion returns the raw generated
text, so without the flag the user gets the model's reasoning instead of a prompt.

These tests are hermetic: the bundled Qwen3 tokenizer ships the real Qwen3 chat template, so no
network access, model weights or GPU are required.
"""

from unittest.mock import MagicMock, patch

import torch
from transformers import PreTrainedTokenizerBase

from invokeai.backend.qwen3.qwen3_tokenizer import load_bundled_qwen3_tokenizer
from invokeai.backend.text_llm_pipeline import TextLLMPipeline

# What the Qwen3 template appends after the generation prompt when thinking is disabled: an
# already-closed, empty thinking block, so the model starts on the answer.
EMPTY_THINKING_BLOCK = "<think>\n\n</think>\n\n"


class _FakeStreamer:
    """Stand-in for TextIteratorStreamer — yields a fixed sequence of text chunks."""

    def __init__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(["expanded ", "prompt"])


def _patch_streamer():
    return patch("invokeai.backend.text_llm_pipeline.TextIteratorStreamer", _FakeStreamer)


class _RecordingTokenizer:
    """Delegates to a real tokenizer while recording what the pipeline sends it."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer
        self.formatted_prompts: list[str] = []
        self.tokenize_kwargs: list[dict] = []

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)

    def __call__(self, text: str, **kwargs):
        self.formatted_prompts.append(text)
        self.tokenize_kwargs.append(kwargs)
        return self._tokenizer(text, **kwargs)


class _FakeEncoding(dict):
    def to(self, *args, **kwargs):
        return self


def _make_system_role_rejecting_tokenizer() -> MagicMock:
    """A Gemma-style tokenizer whose template rejects a dedicated system role."""
    tokenizer = MagicMock()
    tokenizer.chat_template = "<<template>>"

    def apply_chat_template(messages, **kwargs):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "FORMATTED_PROMPT"

    tokenizer.apply_chat_template.side_effect = apply_chat_template
    tokenizer.return_value = _FakeEncoding(
        input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]])
    )
    return tokenizer


def test_thinking_is_disabled_for_a_real_reasoning_chat_template() -> None:
    """The rendered prompt must pre-close the thinking block so generation starts on the answer."""
    tokenizer = _RecordingTokenizer(load_bundled_qwen3_tokenizer())
    pipeline = TextLLMPipeline(MagicMock(), tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a cat", system_prompt="be helpful", max_new_tokens=8)

    assert len(tokenizer.formatted_prompts) == 1
    formatted_prompt = tokenizer.formatted_prompts[0]
    assert "be helpful" in formatted_prompt
    assert "a cat" in formatted_prompt
    assert formatted_prompt.endswith(EMPTY_THINKING_BLOCK)


def test_the_thinking_assertion_is_not_vacuous() -> None:
    """Without the flag the same template leaves the thinking block open — this is the bug."""
    tokenizer = load_bundled_qwen3_tokenizer()

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "a cat"}], tokenize=False, add_generation_prompt=True
    )

    assert not formatted_prompt.endswith(EMPTY_THINKING_BLOCK)


def test_thinking_is_disabled_on_the_system_role_retry() -> None:
    """The Gemma fallback re-renders the template and must keep thinking disabled."""
    tokenizer = _make_system_role_rejecting_tokenizer()
    pipeline = TextLLMPipeline(MagicMock(), tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a cat", system_prompt="be helpful", max_new_tokens=8)

    assert tokenizer.apply_chat_template.call_count == 2
    assert all(call.kwargs["enable_thinking"] is False for call in tokenizer.apply_chat_template.call_args_list)


def test_rendered_chat_template_is_tokenized_without_extra_special_tokens() -> None:
    """The template already emits its control tokens; re-adding them duplicates BOS."""
    tokenizer = _RecordingTokenizer(load_bundled_qwen3_tokenizer())
    pipeline = TextLLMPipeline(MagicMock(), tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a cat", max_new_tokens=8)

    assert tokenizer.tokenize_kwargs[0]["add_special_tokens"] is False


def test_raw_fallback_prompt_is_tokenized_with_special_tokens() -> None:
    """Without a chat template the prompt carries no control tokens, so it still needs them."""
    tokenizer = _RecordingTokenizer(load_bundled_qwen3_tokenizer())
    tokenizer.chat_template = None
    pipeline = TextLLMPipeline(MagicMock(), tokenizer)

    with _patch_streamer():
        pipeline.run(prompt="a cat", max_new_tokens=8)

    assert tokenizer.tokenize_kwargs[0]["add_special_tokens"] is True
