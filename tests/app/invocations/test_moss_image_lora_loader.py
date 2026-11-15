"""The MOSS-Image LoRA nodes: which LoRAs they put on the transformer field, and which they refuse before a generation."""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.invocations.moss_image.moss_image_lora_loader import (
    MossImageLoRACollectionLoader,
    MossImageLoRALoaderInvocation,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

_LORA_NAME = "Pixel Art"


def _context(base: BaseModelType, model_type: ModelType = ModelType.LoRA):
    configs = {"lora": SimpleNamespace(name=_LORA_NAME, base=base, type=model_type)}
    return SimpleNamespace(
        models=SimpleNamespace(exists=lambda key: key in configs, get_config=lambda key: configs[key])
    )


def _lora_field(weight: float = 0.8, key: str = "lora") -> LoRAField:
    return LoRAField(
        lora=ModelIdentifierField(
            key=key, hash="h", name=_LORA_NAME, base=BaseModelType.MossImage, type=ModelType.LoRA
        ),
        weight=weight,
    )


def _transformer_field() -> TransformerField:
    model = ModelIdentifierField(
        key="transformer", hash="h", name="m", base=BaseModelType.MossImage, type=ModelType.Main
    )
    return TransformerField(transformer=model, loras=[])


def _loader(node_class: str, lora: LoRAField):
    if node_class == "single":
        return MossImageLoRALoaderInvocation.model_construct(
            lora=lora.lora, weight=lora.weight, transformer=_transformer_field()
        )
    return MossImageLoRACollectionLoader.model_construct(loras=[lora], transformer=_transformer_field())


def test_the_loader_applies_a_lora_at_its_weight() -> None:
    output = _loader("single", _lora_field(0.6)).invoke(_context(BaseModelType.MossImage))

    assert output.transformer is not None
    assert [(lora.lora.key, lora.weight) for lora in output.transformer.loras] == [("lora", 0.6)]


def test_a_lora_collection_adds_each_lora_once() -> None:
    """Including one an upstream loader applied: a LoRA applied twice doubles its weight."""
    applied = _transformer_field()
    applied.loras.append(_lora_field(0.3, key="upstream"))
    configs = {
        key: SimpleNamespace(name=key, base=BaseModelType.MossImage, type=ModelType.LoRA)
        for key in ("lora", "upstream")
    }
    context = SimpleNamespace(models=SimpleNamespace(exists=configs.__contains__, get_config=configs.__getitem__))

    output = MossImageLoRACollectionLoader.model_construct(
        loras=[_lora_field(0.8), _lora_field(0.5), _lora_field(1.0, key="upstream")], transformer=applied
    ).invoke(context)

    assert output.transformer is not None
    assert [(lora.lora.key, lora.weight) for lora in output.transformer.loras] == [("upstream", 0.3), ("lora", 0.8)]


def test_the_loader_refuses_a_lora_already_on_the_transformer() -> None:
    applied = _transformer_field()
    applied.loras.append(_lora_field(0.3))
    node = MossImageLoRALoaderInvocation.model_construct(lora=_lora_field().lora, weight=1.0, transformer=applied)

    with pytest.raises(ValueError, match="already applied"):
        node.invoke(_context(BaseModelType.MossImage))


@pytest.mark.parametrize("node_class", ["single", "collection"])
def test_a_lora_of_another_base_is_refused_by_the_stored_config_not_the_identifier(node_class: str) -> None:
    """The identifier says MOSS-Image, as a client can make it say; the installed record is a Z-Image LoRA."""
    with pytest.raises(ValueError, match="for z-image models, not MOSS-Image"):
        _loader(node_class, _lora_field()).invoke(_context(BaseModelType.ZImage))


@pytest.mark.parametrize("node_class", ["single", "collection"])
def test_a_lora_that_is_not_installed_is_refused(node_class: str) -> None:
    with pytest.raises(ValueError, match="Unknown LoRA: gone"):
        _loader(node_class, _lora_field(key="gone")).invoke(_context(BaseModelType.MossImage))


@pytest.mark.parametrize("node_class", ["single", "collection"])
def test_a_model_that_is_not_a_lora_is_refused(node_class: str) -> None:
    with pytest.raises(ValueError, match="not a LoRA"):
        _loader(node_class, _lora_field()).invoke(_context(BaseModelType.MossImage, ModelType.Main))
