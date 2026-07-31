"""Regression test for TextLLMPipeline's system-role fallback.

Some chat templates (notably Gemma) reject a dedicated "system" role and raise
"System role not supported". The pipeline should fold the system prompt into the user
turn and retry instead of failing prompt expansion.
"""

from unittest.mock import MagicMock

import torch

from invokeai.backend.text_llm_pipeline import TextLLMPipeline


class _FakeEncoding(dict):
    def to(self, *args, **kwargs):
        return self


def _make_tokenizer(*, rejects_system_role: bool) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.chat_template = "<<template>>"

    def apply_chat_template(messages, **kwargs):
        if rejects_system_role and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "FORMATTED_PROMPT"

    tokenizer.apply_chat_template.side_effect = apply_chat_template
    tokenizer.return_value = _FakeEncoding(
        input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]])
    )
    tokenizer.decode.return_value = "  expanded prompt  "
    return tokenizer


def _make_model() -> MagicMock:
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


def test_system_role_is_folded_into_user_turn_when_unsupported() -> None:
    tokenizer = _make_tokenizer(rejects_system_role=True)
    pipeline = TextLLMPipeline(_make_model(), tokenizer)

    result = pipeline.run("a cat", system_prompt="be helpful", max_new_tokens=8)

    assert result == "expanded prompt"
    # Two attempts: the first with a system role (rejected), the retry with a merged user turn.
    assert tokenizer.apply_chat_template.call_count == 2
    retry_messages = tokenizer.apply_chat_template.call_args_list[1].args[0]
    assert all(m["role"] != "system" for m in retry_messages)
    assert "be helpful" in retry_messages[0]["content"]
    assert "a cat" in retry_messages[0]["content"]


def test_system_role_is_used_directly_when_supported() -> None:
    tokenizer = _make_tokenizer(rejects_system_role=False)
    pipeline = TextLLMPipeline(_make_model(), tokenizer)

    result = pipeline.run("a cat", system_prompt="be helpful", max_new_tokens=8)

    assert result == "expanded prompt"
    # No retry needed: the system role is applied on the first attempt.
    assert tokenizer.apply_chat_template.call_count == 1
    first_messages = tokenizer.apply_chat_template.call_args_list[0].args[0]
    assert first_messages[0]["role"] == "system"
