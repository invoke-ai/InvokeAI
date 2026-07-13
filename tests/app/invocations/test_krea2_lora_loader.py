from types import SimpleNamespace

from invokeai.app.invocations.krea2_lora_loader import Krea2LoRACollectionLoader, Krea2LoRALoaderInvocation
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, Qwen3VLEncoderField, TransformerField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


def _model(key: str, model_type: ModelType, base: BaseModelType = BaseModelType.Krea2) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _lora(key: str = "lora") -> LoRAField:
    return LoRAField(lora=_model(key, ModelType.LoRA), weight=1.0)


def _transformer(loras: list[LoRAField]) -> TransformerField:
    return TransformerField(
        transformer=_model("main", ModelType.Main).model_copy(update={"submodel_type": SubModelType.Transformer}),
        loras=loras,
    )


def _encoder(loras: list[LoRAField]) -> Qwen3VLEncoderField:
    encoder = _model("encoder", ModelType.Qwen3VLEncoder, BaseModelType.Any)
    return Qwen3VLEncoderField(
        tokenizer=encoder.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=loras,
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(models=SimpleNamespace(exists=lambda _key: True))


def test_collection_loader_repairs_transformer_only_lora_state() -> None:
    lora = _lora()
    existing = lora.model_copy(update={"weight": 0.25})
    invocation = Krea2LoRACollectionLoader.model_construct(
        loras=[lora], transformer=_transformer([existing]), qwen3_vl_encoder=_encoder([])
    )

    output = invocation.invoke(_context())

    assert [item.lora.key for item in output.transformer.loras] == ["lora"]
    assert [item.lora.key for item in output.qwen3_vl_encoder.loras] == ["lora"]
    assert output.qwen3_vl_encoder.loras[0].weight == 0.25


def test_collection_loader_repairs_encoder_only_lora_state() -> None:
    lora = _lora()
    invocation = Krea2LoRACollectionLoader.model_construct(
        loras=[lora], transformer=_transformer([]), qwen3_vl_encoder=_encoder([lora])
    )

    output = invocation.invoke(_context())

    assert [item.lora.key for item in output.transformer.loras] == ["lora"]
    assert [item.lora.key for item in output.qwen3_vl_encoder.loras] == ["lora"]


def test_collection_loader_does_not_duplicate_synchronized_lora_state() -> None:
    lora = _lora()
    invocation = Krea2LoRACollectionLoader.model_construct(
        loras=[lora], transformer=_transformer([lora]), qwen3_vl_encoder=_encoder([lora])
    )

    output = invocation.invoke(_context())

    assert len(output.transformer.loras) == 1
    assert len(output.qwen3_vl_encoder.loras) == 1


def test_single_loader_repairs_transformer_only_lora_state() -> None:
    lora = _lora()
    existing = lora.model_copy(update={"weight": 0.25})
    invocation = Krea2LoRALoaderInvocation.model_construct(
        lora=lora.lora, weight=lora.weight, transformer=_transformer([existing]), qwen3_vl_encoder=_encoder([])
    )

    output = invocation.invoke(_context())

    assert len(output.transformer.loras) == 1
    assert [item.lora.key for item in output.qwen3_vl_encoder.loras] == ["lora"]
    assert output.qwen3_vl_encoder.loras[0].weight == 0.25


def test_single_loader_rejects_non_krea_lora() -> None:
    lora = _model("flux-lora", ModelType.LoRA, BaseModelType.Flux)
    invocation = Krea2LoRALoaderInvocation.model_construct(
        lora=lora, weight=1.0, transformer=_transformer([]), qwen3_vl_encoder=_encoder([])
    )

    try:
        invocation.invoke(_context())
    except ValueError as error:
        assert "not Krea-2 models" in str(error)
    else:
        raise AssertionError("Expected a non-Krea LoRA to be rejected")
