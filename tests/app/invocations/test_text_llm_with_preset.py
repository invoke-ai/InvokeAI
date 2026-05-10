"""Lightweight test for TextLLMWithPresetInvocation.

The model-loading and pipeline-run code paths are exercised by the existing
TextLLMInvocation node. The only behaviour unique to TextLLMWithPresetInvocation
is that the system prompt is fetched from system_prompt_records by id; we mock
the shared LLM helper and assert the lookup happens with the expected content.
"""

from unittest.mock import MagicMock, patch

import pytest

from invokeai.app.invocations.fields import SystemPromptField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.text_llm import TextLLMWithPresetInvocation
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptNotFoundError,
)


def _make_invocation(prompt_id: str = "test-id") -> TextLLMWithPresetInvocation:
    return TextLLMWithPresetInvocation(
        id="test-node",
        prompt="a cat",
        system_prompt=SystemPromptField(system_prompt_id=prompt_id),
        text_llm_model=ModelIdentifierField(
            key="dummy", hash="x", name="dummy", base="any", type="text_llm"
        ),
        max_tokens=50,
    )


def _make_context(prompt_record_content: str | None) -> MagicMock:
    """Build a context whose system_prompt_records.get returns a record with the given content,
    or raises SystemPromptNotFoundError when content is None."""
    context = MagicMock()
    if prompt_record_content is None:
        context._services.system_prompt_records.get.side_effect = SystemPromptNotFoundError("not found")
    else:
        record = MagicMock()
        record.content = prompt_record_content
        context._services.system_prompt_records.get.return_value = record
    return context


def test_preset_node_loads_content_from_db_and_passes_to_llm() -> None:
    inv = _make_invocation(prompt_id="abc")
    context = _make_context(prompt_record_content="custom system instruction")

    with patch("invokeai.app.invocations.text_llm._run_text_llm", return_value="expanded") as mock_run:
        result = inv.invoke(context)

    # The DB lookup happened with the configured id …
    context._services.system_prompt_records.get.assert_called_once_with("abc")
    # … and the content from the record was forwarded to the shared pipeline helper.
    assert mock_run.call_count == 1
    kwargs = mock_run.call_args.kwargs
    assert kwargs["system_prompt"] == "custom system instruction"
    assert kwargs["prompt"] == "a cat"
    assert kwargs["max_tokens"] == 50

    assert result.value == "expanded"


def test_preset_node_propagates_not_found_error() -> None:
    inv = _make_invocation(prompt_id="missing-id")
    context = _make_context(prompt_record_content=None)

    with patch("invokeai.app.invocations.text_llm._run_text_llm") as mock_run:
        with pytest.raises(SystemPromptNotFoundError):
            inv.invoke(context)

    # Pipeline must NOT be invoked if the preset can't be resolved.
    mock_run.assert_not_called()
