"""Tests for TextLLMWithPresetInvocation.

The model-loading and pipeline-run code paths are exercised by the existing
TextLLMInvocation node. What is unique to TextLLMWithPresetInvocation is that the system
prompt is fetched from system_prompt_records by id -- and that this lookup must enforce the
same own/public/default/admin rules the REST API does, since the record store itself is
unscoped and the resolved content ends up in the LLM's system message.
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
        text_llm_model=ModelIdentifierField(key="dummy", hash="x", name="dummy", base="any", type="text_llm"),
        max_tokens=50,
    )


def _make_context(
    prompt_record_content: str | None,
    *,
    multiuser: bool = False,
    record_user_id: str = "system",
    record_is_public: bool = True,
    queue_user_id: str = "alice",
    queue_user_is_admin: bool = False,
) -> MagicMock:
    """Build a context whose system_prompt_records.get returns a record with the given content,
    or raises SystemPromptNotFoundError when content is None."""
    context = MagicMock()
    context._services.configuration.multiuser = multiuser
    context._data.queue_item.user_id = queue_user_id
    context._services.users.get.return_value = MagicMock(is_admin=queue_user_is_admin)
    if prompt_record_content is None:
        context._services.system_prompt_records.get.side_effect = SystemPromptNotFoundError("not found")
    else:
        record = MagicMock()
        record.content = prompt_record_content
        record.user_id = record_user_id
        record.is_public = record_is_public
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


def test_preset_node_reports_a_missing_preset() -> None:
    inv = _make_invocation(prompt_id="missing-id")
    context = _make_context(prompt_record_content=None)

    with patch("invokeai.app.invocations.text_llm._run_text_llm") as mock_run:
        with pytest.raises(ValueError, match="could not be found"):
            inv.invoke(context)

    # Pipeline must NOT be invoked if the preset can't be resolved.
    mock_run.assert_not_called()


def test_preset_node_rejects_another_users_private_prompt_in_multiuser() -> None:
    # The exact escalation the REST layer blocks: user2 knows the id of user1's private prompt
    # (the create response echoes it) and enqueues a graph referencing it.
    inv = _make_invocation(prompt_id="alice-private")
    context = _make_context(
        prompt_record_content="alice's secret instruction",
        multiuser=True,
        record_user_id="alice",
        record_is_public=False,
        queue_user_id="bob",
    )

    with patch("invokeai.app.invocations.text_llm._run_text_llm") as mock_run:
        with pytest.raises(ValueError, match="not accessible to this user"):
            inv.invoke(context)

    mock_run.assert_not_called()


@pytest.mark.parametrize(
    "record_user_id,record_is_public,queue_user_id,queue_user_is_admin",
    [
        ("system", False, "bob", False),  # seeded default
        ("alice", True, "bob", False),  # explicitly shared
        ("bob", False, "bob", False),  # owner
        ("alice", False, "admin", True),  # admin
    ],
)
def test_preset_node_allows_accessible_prompts_in_multiuser(
    record_user_id: str, record_is_public: bool, queue_user_id: str, queue_user_is_admin: bool
) -> None:
    inv = _make_invocation(prompt_id="some-id")
    context = _make_context(
        prompt_record_content="instruction",
        multiuser=True,
        record_user_id=record_user_id,
        record_is_public=record_is_public,
        queue_user_id=queue_user_id,
        queue_user_is_admin=queue_user_is_admin,
    )

    with patch("invokeai.app.invocations.text_llm._run_text_llm", return_value="expanded") as mock_run:
        assert inv.invoke(context).value == "expanded"

    assert mock_run.call_args.kwargs["system_prompt"] == "instruction"


def test_preset_node_skips_the_ownership_check_in_single_user_mode() -> None:
    # Single-user installs have no meaningful user identity; every prompt is reachable.
    inv = _make_invocation(prompt_id="some-id")
    context = _make_context(
        prompt_record_content="instruction",
        multiuser=False,
        record_user_id="someone-else",
        record_is_public=False,
    )

    with patch("invokeai.app.invocations.text_llm._run_text_llm", return_value="expanded") as mock_run:
        assert inv.invoke(context).value == "expanded"

    context._services.users.get.assert_not_called()
    assert mock_run.call_count == 1
