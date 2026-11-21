from types import SimpleNamespace

import pytest

from invokeai.app.invocations.model import CLIPField, LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.invocations.sd3.sd3_lora_loader import (
    SD3LoRACollectionLoader,
    SD3LoRALoaderInvocation,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


def _model(
    key: str, model_type: ModelType, base: BaseModelType = BaseModelType.StableDiffusion3
) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _lora(key: str = "lora", weight: float = 1.0) -> LoRAField:
    return LoRAField(lora=_model(key, ModelType.LoRA), weight=weight)


def _transformer(loras: list[LoRAField]) -> TransformerField:
    return TransformerField(
        transformer=_model("main", ModelType.Main).model_copy(update={"submodel_type": SubModelType.Transformer}),
        loras=loras,
    )


def _context(stored_base: BaseModelType = BaseModelType.StableDiffusion3) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(
            exists=lambda _key: True,
            get_config=lambda _key: SimpleNamespace(base=stored_base, type=ModelType.LoRA),
        )
    )


def test_the_loader_appends_the_lora_and_echoes_it_for_metadata() -> None:
    invocation = SD3LoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=0.6, transformer=_transformer([])
    )

    output = invocation.invoke(_context())

    assert [(item.lora.key, item.weight) for item in output.transformer.loras] == [("lora", 0.6)]
    assert (output.lora_metadata.model.key, output.lora_metadata.weight) == ("lora", 0.6)
    # The input field is not mutated: the graph may hand the same transformer to another branch.
    assert invocation.transformer.loras == []


def test_the_loader_refuses_a_lora_applied_twice() -> None:
    invocation = SD3LoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([_lora()])
    )

    with pytest.raises(ValueError, match="already applied"):
        invocation.invoke(_context())


@pytest.mark.parametrize("node", ["single", "collection"])
def test_a_lora_of_another_family_is_refused_whatever_the_identifier_claims(node: str) -> None:
    """The identifier is client-supplied; only the stored config decides what gets patched in."""
    if node == "single":
        invocation = SD3LoRALoaderInvocation.model_construct(
            lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([])
        )
    else:
        invocation = SD3LoRACollectionLoader.model_construct(loras=[_lora()], transformer=_transformer([]))

    with pytest.raises(ValueError, match="not an SD 3 LoRA"):
        invocation.invoke(_context(stored_base=BaseModelType.QwenImage))


def test_the_collection_loader_skips_duplicates_and_reports_what_it_applied() -> None:
    first, second = _lora("first", 0.5), _lora("second", 0.9)
    invocation = SD3LoRACollectionLoader.model_construct(
        loras=[first, second, first], transformer=_transformer([first])
    )

    output = invocation.invoke(_context())

    assert [item.lora.key for item in output.transformer.loras] == ["first", "second"]
    assert [(item.model.key, item.weight) for item in output.lora_metadata] == [("second", 0.9)]


def test_the_collection_loader_without_a_transformer_outputs_nothing() -> None:
    invocation = SD3LoRACollectionLoader.model_construct(loras=[_lora()], transformer=None)

    output = invocation.invoke(_context())

    assert output.transformer is None
    assert output.lora_metadata == []


def _clip(loras: list[LoRAField]) -> CLIPField:
    return CLIPField(
        tokenizer=_model("main", ModelType.Main).model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=_model("main", ModelType.Main).model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=loras,
        skipped_layers=0,
    )


@pytest.mark.parametrize("node", ["single", "collection"])
def test_the_lora_reaches_the_transformer_and_both_clips(node: str) -> None:
    """An SD 3 LoRA may train CLIP-L and CLIP-G as well; the encoders pick their own layers out of it."""
    if node == "single":
        invocation = SD3LoRALoaderInvocation.model_construct(
            lora=_model("lora", ModelType.LoRA),
            weight=0.6,
            transformer=_transformer([]),
            clip_l=_clip([]),
            clip_g=_clip([]),
        )
    else:
        invocation = SD3LoRACollectionLoader.model_construct(
            loras=[_lora(weight=0.6)], transformer=_transformer([]), clip_l=_clip([]), clip_g=_clip([])
        )

    output = invocation.invoke(_context())

    for field in (output.transformer, output.clip_l, output.clip_g):
        assert [(item.lora.key, item.weight) for item in field.loras] == [("lora", 0.6)]
