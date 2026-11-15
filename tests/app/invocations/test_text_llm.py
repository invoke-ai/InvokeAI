"""The Text LLM nodes take the model's tokenizer from the model loader.

Read from the model's path, the tokenizer of a single-file Text LLM (the FLUX.3 prompt upsampler's ComfyUI build)
cannot load: there is no folder. Its loader serves a vendored one, as a submodel, and both nodes run through
`_run_text_llm`.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from invokeai.app.invocations import text_llm as text_llm_module
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.text_llm import TextLLMInvocation
from invokeai.backend.model_manager.taxonomy import SubModelType


class _RecordingPipeline:
    """Stands in for `TextLLMPipeline`, which would generate: records the tokenizer it was handed."""

    received: list[Any] = []

    def __init__(self, model: Any, tokenizer: Any, processor: Any = None) -> None:
        _RecordingPipeline.received.append(tokenizer)

    def run(self, **kwargs: Any) -> str:
        return "expanded"


def test_the_node_takes_its_tokenizer_from_the_model_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    requested: list[SubModelType | None] = []

    def load(identifier: ModelIdentifierField, submodel_type: SubModelType | None = None) -> Any:
        requested.append(submodel_type)
        loaded = MagicMock()
        if submodel_type is SubModelType.Tokenizer:
            loaded.model = tokenizer
        else:
            loaded.model_on_device.return_value.__enter__.return_value = (None, torch.nn.Linear(1, 1))
        return loaded

    context = MagicMock()
    context.models.load.side_effect = load
    monkeypatch.setattr(text_llm_module, "TextLLMPipeline", _RecordingPipeline)
    _RecordingPipeline.received.clear()
    node = TextLLMInvocation(
        id="text-llm",
        prompt="a fox",
        text_llm_model=ModelIdentifierField(key="pup", hash="x", name="pup", base="any", type="text_llm"),
        seed=1,
    )

    result = node.invoke(context)

    assert result.value == "expanded"
    # The tokenizer first, so reading it does not hold the model on the device; nothing read from a path.
    assert requested == [SubModelType.Tokenizer, None]
    assert _RecordingPipeline.received == [tokenizer]
    context.models.get_absolute_path.assert_not_called()
