"""Tests for the MiniMax H3 LoRA loaders' metadata echo.

Regression context: the single loader gained a required ``lora_metadata`` output so a
workflow can record the LoRA it applied without retyping the model key. The collection
loader shared that output class and constructed it with no arguments, so every
invocation of ``minimax_h3_lora_collection_loader`` -- which the legacy Generate tab
inserts whenever any H3 LoRA is enabled -- raised a pydantic ValidationError. It now has
its own output whose echo is a list, the shape ``core_metadata.loras`` takes directly.
"""

from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.minimax_h3.minimax_h3_lora_loader import (
    MiniMaxH3LoRACollectionLoader,
    MiniMaxH3LoRACollectionLoaderOutput,
    MiniMaxH3LoRALoaderInvocation,
)
from invokeai.app.invocations.model import LoRAField, MiniMaxH3TransformerField, ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _identifier(key: str, model_type: ModelType = ModelType.LoRA) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=BaseModelType.MiniMaxH3, type=model_type)


def _transformer() -> MiniMaxH3TransformerField:
    return MiniMaxH3TransformerField(transformer=_identifier("transformer", ModelType.Main))


def _context(*lora_keys: str) -> MagicMock:
    context = MagicMock()
    context.models.exists.side_effect = lambda key: key in lora_keys
    context.models.get_config.side_effect = lambda key: MagicMock(type=ModelType.LoRA, base=BaseModelType.MiniMaxH3)
    return context


def test_collection_loader_output_constructs_with_no_arguments() -> None:
    """The exact regression: `invoke` builds this empty on the no-transformer path."""
    output = MiniMaxH3LoRACollectionLoaderOutput()
    assert output.transformer is None
    assert output.lora_metadata == []


def test_collection_loader_returns_cleanly_without_a_transformer() -> None:
    output = MiniMaxH3LoRACollectionLoader(id="n").invoke(_context())
    assert output.transformer is None
    assert output.lora_metadata == []


def test_collection_loader_echoes_every_applied_lora() -> None:
    loras = [LoRAField(lora=_identifier("turbo"), weight=1.0), LoRAField(lora=_identifier("style"), weight=0.6)]
    node = MiniMaxH3LoRACollectionLoader(id="n", transformer=_transformer(), loras=loras)

    output = node.invoke(_context("turbo", "style"))

    assert output.transformer is not None
    assert [item.lora.key for item in output.transformer.loras] == ["turbo", "style"]
    assert [(item.model.key, item.weight) for item in output.lora_metadata] == [("turbo", 1.0), ("style", 0.6)]


def test_collection_loader_skips_duplicates_in_both_the_transformer_and_the_echo() -> None:
    transformer = _transformer()
    transformer.loras.append(LoRAField(lora=_identifier("turbo"), weight=1.0))
    node = MiniMaxH3LoRACollectionLoader(
        id="n", transformer=transformer, loras=[LoRAField(lora=_identifier("turbo"), weight=0.5)]
    )

    output = node.invoke(_context("turbo"))

    assert output.transformer is not None
    assert len(output.transformer.loras) == 1
    assert output.lora_metadata == []


def test_collection_loader_accepts_a_single_lora() -> None:
    node = MiniMaxH3LoRACollectionLoader(
        id="n", transformer=_transformer(), loras=LoRAField(lora=_identifier("turbo"), weight=1.0)
    )

    output = node.invoke(_context("turbo"))

    assert [item.model.key for item in output.lora_metadata] == ["turbo"]


def test_collection_loader_does_not_mutate_its_input_transformer() -> None:
    transformer = _transformer()
    node = MiniMaxH3LoRACollectionLoader(
        id="n", transformer=transformer, loras=[LoRAField(lora=_identifier("turbo"), weight=1.0)]
    )

    node.invoke(_context("turbo"))

    assert transformer.loras == []


def test_single_loader_echoes_the_lora_it_applied() -> None:
    node = MiniMaxH3LoRALoaderInvocation(id="n", lora=_identifier("turbo"), weight=0.8, transformer=_transformer())

    output = node.invoke(_context("turbo"))

    assert output.lora_metadata.model.key == "turbo"
    assert output.lora_metadata.weight == 0.8
    assert output.transformer is not None
    assert [item.lora.key for item in output.transformer.loras] == ["turbo"]


def test_single_loader_rejects_a_non_h3_lora() -> None:
    node = MiniMaxH3LoRALoaderInvocation(id="n", lora=_identifier("turbo"), transformer=_transformer())
    context = _context("turbo")
    context.models.get_config.side_effect = lambda key: MagicMock(type=ModelType.LoRA, base=BaseModelType.Flux)

    with pytest.raises(ValueError, match="is not a MiniMax H3 LoRA"):
        node.invoke(context)
