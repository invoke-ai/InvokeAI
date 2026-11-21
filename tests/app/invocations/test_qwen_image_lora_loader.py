from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, QwenVLEncoderField, TransformerField
from invokeai.app.invocations.qwen_image.qwen_image_lora_loader import (
    QwenImageLoRACollectionLoader,
    QwenImageLoRALoaderInvocation,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    lora_model_from_qwen_image_state_dict,
)


def _model(key: str, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=BaseModelType.QwenImage, type=model_type)


def _lora(key: str = "lora", weight: float = 1.0) -> LoRAField:
    return LoRAField(lora=_model(key, ModelType.LoRA), weight=weight)


def _transformer(loras: list[LoRAField]) -> TransformerField:
    return TransformerField(
        transformer=_model("main", ModelType.Main).model_copy(update={"submodel_type": SubModelType.Transformer}),
        loras=loras,
    )


def _encoder(loras: list[LoRAField]) -> QwenVLEncoderField:
    encoder = _model("encoder", ModelType.QwenVLEncoder)
    return QwenVLEncoderField(
        tokenizer=encoder.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=loras,
    )


def _patch(*layers: str):
    state_dict = {}
    for layer in layers:
        state_dict[f"{layer}.lora_A.weight"] = torch.zeros(2, 4)
        state_dict[f"{layer}.lora_B.weight"] = torch.zeros(4, 2)
    return lora_model_from_qwen_image_state_dict(state_dict)


TRANSFORMER_ONLY = ("transformer_blocks.0.attn.to_k",)
WITH_ENCODER_LAYERS = (*TRANSFORMER_ONLY, "text_encoder.model.language_model.layers.0.mlp.down_proj")


def _context(layers: tuple[str, ...] = TRANSFORMER_ONLY) -> SimpleNamespace:
    warnings: list[str] = []
    return SimpleNamespace(
        models=SimpleNamespace(
            exists=lambda _key: True,
            load=lambda _identifier: SimpleNamespace(model=_patch(*layers)),
            get_config=lambda _key: SimpleNamespace(name="My LoRA"),
        ),
        logger=SimpleNamespace(warning=warnings.append),
        warnings=warnings,
    )


def test_the_encoder_field_defaults_to_no_loras() -> None:
    """Saved graphs carry encoder fields without `loras`."""
    encoder = _model("encoder", ModelType.QwenVLEncoder)

    assert QwenVLEncoderField(tokenizer=encoder, text_encoder=encoder).loras == []


def test_single_loader_adds_the_lora_to_transformer_and_encoder_with_the_weight() -> None:
    invocation = QwenImageLoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=0.6, transformer=_transformer([]), qwen_vl_encoder=_encoder([])
    )

    output = invocation.invoke(_context())

    assert [(item.lora.key, item.weight) for item in output.transformer.loras] == [("lora", 0.6)]
    assert [(item.lora.key, item.weight) for item in output.qwen_vl_encoder.loras] == [("lora", 0.6)]


def test_single_loader_leaves_its_inputs_untouched() -> None:
    transformer, encoder = _transformer([]), _encoder([])
    invocation = QwenImageLoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=transformer, qwen_vl_encoder=encoder
    )

    invocation.invoke(_context())

    assert transformer.loras == [] and encoder.loras == []


def test_a_transformer_only_graph_still_works() -> None:
    invocation = QwenImageLoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([]), qwen_vl_encoder=None
    )

    output = invocation.invoke(_context())

    assert [item.lora.key for item in output.transformer.loras] == ["lora"]
    assert output.qwen_vl_encoder is None


def test_an_encoder_only_graph_works() -> None:
    invocation = QwenImageLoRALoaderInvocation.model_construct(
        lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=None, qwen_vl_encoder=_encoder([])
    )

    output = invocation.invoke(_context())

    assert output.transformer is None
    assert [item.lora.key for item in output.qwen_vl_encoder.loras] == ["lora"]


@pytest.mark.parametrize("applied_to", ["transformer", "encoder"])
def test_single_loader_rejects_a_lora_already_applied(applied_to: str) -> None:
    lora = _lora()
    invocation = QwenImageLoRALoaderInvocation.model_construct(
        lora=lora.lora,
        weight=1.0,
        transformer=_transformer([lora] if applied_to == "transformer" else []),
        qwen_vl_encoder=_encoder([lora] if applied_to == "encoder" else []),
    )

    with pytest.raises(ValueError, match="already applied"):
        invocation.invoke(_context())


def test_collection_loader_adds_each_lora_once_to_both_models() -> None:
    first, second = _lora("a", 0.5), _lora("b", 0.8)
    invocation = QwenImageLoRACollectionLoader.model_construct(
        loras=[first, second, first], transformer=_transformer([]), qwen_vl_encoder=_encoder([])
    )

    output = invocation.invoke(_context())

    expected = [("a", 0.5), ("b", 0.8)]
    assert [(item.lora.key, item.weight) for item in output.transformer.loras] == expected
    assert [(item.lora.key, item.weight) for item in output.qwen_vl_encoder.loras] == expected


def test_collection_loader_works_without_the_encoder() -> None:
    invocation = QwenImageLoRACollectionLoader.model_construct(
        loras=_lora(), transformer=_transformer([]), qwen_vl_encoder=None
    )

    output = invocation.invoke(_context())

    assert [item.lora.key for item in output.transformer.loras] == ["lora"]
    assert output.qwen_vl_encoder is None


@pytest.mark.parametrize("loader", ["single", "collection"])
def test_a_transformer_only_connection_warns_once_when_the_lora_has_encoder_layers(loader: str) -> None:
    """Workflows saved before the encoder connection existed would otherwise lose the layers without a trace."""
    context = _context(WITH_ENCODER_LAYERS)
    if loader == "single":
        invocation = QwenImageLoRALoaderInvocation.model_construct(
            lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([]), qwen_vl_encoder=None
        )
    else:
        invocation = QwenImageLoRACollectionLoader.model_construct(
            loras=[_lora(), _lora()], transformer=_transformer([]), qwen_vl_encoder=None
        )

    invocation.invoke(context)

    assert len(context.warnings) == 1
    assert "My LoRA" in context.warnings[0] and "Qwen VL Encoder" in context.warnings[0]


@pytest.mark.parametrize("loader", ["single", "collection"])
@pytest.mark.parametrize(
    ("layers", "connect_encoder"),
    [(TRANSFORMER_ONLY, False), (WITH_ENCODER_LAYERS, True)],
    ids=["transformer_only_lora", "encoder_connected"],
)
def test_no_warning_without_unused_encoder_layers(loader: str, layers: tuple[str, ...], connect_encoder: bool) -> None:
    context = _context(layers)
    encoder = _encoder([]) if connect_encoder else None
    if loader == "single":
        invocation = QwenImageLoRALoaderInvocation.model_construct(
            lora=_model("lora", ModelType.LoRA), weight=1.0, transformer=_transformer([]), qwen_vl_encoder=encoder
        )
    else:
        invocation = QwenImageLoRACollectionLoader.model_construct(
            loras=[_lora()], transformer=_transformer([]), qwen_vl_encoder=encoder
        )

    invocation.invoke(context)

    assert context.warnings == []
