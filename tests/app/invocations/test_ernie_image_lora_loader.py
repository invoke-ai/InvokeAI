from types import SimpleNamespace

import pytest

from invokeai.app.invocations.ernie_image.ernie_image_lora_loader import (
    ErnieImageLoRACollectionLoader,
    ErnieImageLoRALoaderInvocation,
)
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, TransformerField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


def _model(key: str, model_type: ModelType, base: BaseModelType = BaseModelType.ErnieImage) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _lora(key: str = "lora", weight: float = 1.0) -> LoRAField:
    return LoRAField(lora=_model(key, ModelType.LoRA), weight=weight)


def _transformer(loras: list[LoRAField]) -> TransformerField:
    return TransformerField(
        transformer=_model("main", ModelType.Main).model_copy(update={"submodel_type": SubModelType.Transformer}),
        loras=loras,
    )


def _context(stored_base: BaseModelType = BaseModelType.ErnieImage) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(
            exists=lambda _key: True,
            get_config=lambda _key: SimpleNamespace(base=stored_base, type=ModelType.LoRA),
        )
    )


def test_the_loader_appends_the_lora_and_echoes_it_for_metadata() -> None:
    invocation = ErnieImageLoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=0.6, transformer=_transformer([])
    )

    output = invocation.invoke(_context())

    assert [(item.lora.key, item.weight) for item in output.transformer.loras] == [("lora", 0.6)]
    assert (output.lora_metadata.model.key, output.lora_metadata.weight) == ("lora", 0.6)
    # The input field is not mutated: the graph may hand the same transformer to another branch.
    assert invocation.transformer.loras == []


def test_the_loader_refuses_a_lora_applied_twice() -> None:
    invocation = ErnieImageLoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([_lora()])
    )

    with pytest.raises(ValueError, match="already applied"):
        invocation.invoke(_context())


@pytest.mark.parametrize("node", ["single", "collection"])
def test_a_lora_of_another_family_is_refused_whatever_the_identifier_claims(node: str) -> None:
    """The identifier is client-supplied; only the stored config decides what gets patched in."""
    if node == "single":
        invocation = ErnieImageLoRALoaderInvocation.model_construct(
            lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([])
        )
    else:
        invocation = ErnieImageLoRACollectionLoader.model_construct(loras=[_lora()], transformer=_transformer([]))

    with pytest.raises(ValueError, match="not an ERNIE-Image LoRA"):
        invocation.invoke(_context(stored_base=BaseModelType.ZImage))


def test_the_collection_loader_skips_duplicates_and_reports_what_it_applied() -> None:
    first, second = _lora("first", 0.5), _lora("second", 0.9)
    invocation = ErnieImageLoRACollectionLoader.model_construct(
        loras=[first, second, first], transformer=_transformer([first])
    )

    output = invocation.invoke(_context())

    assert [item.lora.key for item in output.transformer.loras] == ["first", "second"]
    assert [(item.model.key, item.weight) for item in output.lora_metadata] == [("second", 0.9)]


def test_the_collection_loader_without_a_transformer_outputs_nothing() -> None:
    invocation = ErnieImageLoRACollectionLoader.model_construct(loras=[_lora()], transformer=None)

    output = invocation.invoke(_context())

    assert output.transformer is None
    assert output.lora_metadata == []
