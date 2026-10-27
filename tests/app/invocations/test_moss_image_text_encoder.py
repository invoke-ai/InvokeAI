"""`moss_image_text_encoder` refuses an encoder other than Qwen3.5 9B before loading it.

The model loader checks its own selection, but in the workflow editor this node can be wired to any Qwen3.5 encoder
output -- Anima's loader hands out the 4B.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField, Qwen35EncoderField
from invokeai.app.invocations.text_encoder.moss_image_text_encoder import MossImageTextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen35VariantType,
    SubModelType,
)


class _LoadReached(Exception):
    """Raised by the fake model cache: the node got past its check to loading the encoder."""


def _context(config: SimpleNamespace) -> MagicMock:
    context = MagicMock()
    context.models.get_config.return_value = config
    context.models.load.side_effect = _LoadReached
    return context


def _invoke(context: MagicMock, config: SimpleNamespace) -> None:
    identifier = ModelIdentifierField(
        key="encoder", hash="encoder", name=config.name, base=config.base, type=config.type
    )
    invocation = MossImageTextEncoderInvocation(
        prompt="a cat",
        qwen3_5_encoder=Qwen35EncoderField(
            tokenizer=identifier.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
            text_encoder=identifier.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        ),
    )
    invocation.invoke(context)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            name="qwen35_4b", base=BaseModelType.Any, type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_4B
        ),
        SimpleNamespace(
            name="Anima-3.8B", base=BaseModelType.Anima, type=ModelType.Main, format=ModelFormat.Checkpoint
        ),
    ],
    ids=["qwen3_5_4b", "another_main_model"],
)
def test_an_encoder_other_than_the_9b_is_refused_before_it_loads(config: SimpleNamespace) -> None:
    context = _context(config)

    with pytest.raises(ValueError, match=f"'{config.name}' is not a Qwen3.5 9B encoder"):
        _invoke(context, config)
    context.models.load.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(
            name="qwen35_9b", base=BaseModelType.Any, type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_9B
        ),
        SimpleNamespace(
            name="MOSS-Image Base", base=BaseModelType.MossImage, type=ModelType.Main, format=ModelFormat.Diffusers
        ),
    ],
    ids=["standalone_9b", "moss_image_pipeline"],
)
def test_the_9b_and_the_pipeline_s_own_encoder_go_on_to_load(config: SimpleNamespace) -> None:
    with pytest.raises(_LoadReached):
        _invoke(_context(config), config)
