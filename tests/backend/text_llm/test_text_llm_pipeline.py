"""Tests for the TextLLMPipeline class."""

from unittest.mock import MagicMock

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

    # Simulate tokenizer __call__ returning dict with input_ids
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer_output = MagicMock()
    tokenizer_output.__getitem__ = lambda self, key: {"input_ids": input_ids}[key]
    tokenizer_output.to.return_value = tokenizer_output
    tokenizer.return_value = tokenizer_output

    tokenizer.decode.return_value = "A detailed landscape with mountains"
    return tokenizer


def _make_mock_model() -> MagicMock:
    """Create a mock causal LM model."""
    model = MagicMock()
    # generate returns tensor that includes input + generated tokens
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])
    return model


def test_pipeline_uses_chat_template_when_available():
    """Pipeline should use apply_chat_template when the tokenizer supports it."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

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

    pipeline.run(prompt="a cat", system_prompt="Be helpful", device=torch.device("cpu"), dtype=torch.float32)

    tokenizer.apply_chat_template.assert_not_called()
    # Check that the tokenizer was called with the fallback format
    call_args = tokenizer.call_args[0][0]
    assert "Be helpful" in call_args
    assert "a cat" in call_args
    assert "Assistant:" in call_args


def test_pipeline_no_system_prompt():
    """Pipeline should work without a system prompt."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    pipeline.run(prompt="a dog", system_prompt="", device=torch.device("cpu"), dtype=torch.float32)

    call_args = tokenizer.apply_chat_template.call_args
    messages = call_args[0][0]
    # No system message when system_prompt is empty
    assert not any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" and m["content"] == "a dog" for m in messages)


def test_pipeline_decodes_only_generated_tokens():
    """Pipeline should strip input tokens and only decode newly generated ones."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    pipeline.run(prompt="test", device=torch.device("cpu"), dtype=torch.float32)

    # The mock model returns [1,2,3,4,5,10,11,12], input is [1,2,3,4,5]
    # So decode should be called with [10, 11, 12]
    decode_call = tokenizer.decode.call_args
    decoded_tokens = decode_call[0][0]
    assert decoded_tokens.tolist() == [10, 11, 12]
    assert decode_call[1]["skip_special_tokens"] is True


def test_pipeline_passes_generation_params():
    """Pipeline should pass max_new_tokens and sampling params to model.generate."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    pipeline.run(prompt="test", max_new_tokens=100, device=torch.device("cpu"), dtype=torch.float32)

    generate_kwargs = model.generate.call_args[1]
    assert generate_kwargs["max_new_tokens"] == 100
    assert generate_kwargs["do_sample"] is True
    assert generate_kwargs["temperature"] == 0.7
    assert generate_kwargs["top_p"] == 0.9


def test_pipeline_returns_stripped_string():
    """Pipeline should return a stripped string from the decoded output."""
    tokenizer = _make_mock_tokenizer(has_chat_template=True)
    tokenizer.decode.return_value = "  generated text with spaces  "
    model = _make_mock_model()
    pipeline = TextLLMPipeline(model, tokenizer)

    result = pipeline.run(prompt="test", device=torch.device("cpu"), dtype=torch.float32)

    assert result == "generated text with spaces"


def test_default_system_prompt_content():
    """The default system prompt should mention image generation."""
    assert "image generation" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "prompt" in DEFAULT_SYSTEM_PROMPT.lower()
