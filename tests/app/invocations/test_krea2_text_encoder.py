from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.krea2_text_encoder import Krea2TextEncoderInvocation
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, Qwen3VLEncoderField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_QWEN3VL_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


class _Tokenizer:
    def __call__(self, _text, **_kwargs):
        return SimpleNamespace(
            input_ids=torch.ones((1, 40), dtype=torch.long),
            attention_mask=torch.ones((1, 40), dtype=torch.long),
        )


class _TextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **_kwargs):
        hidden_states = tuple(torch.full((1, 40, 4), float(index)) for index in range(36))
        return SimpleNamespace(hidden_states=hidden_states)


class _TokenizerInfo:
    def __enter__(self):
        return _Tokenizer()

    def __exit__(self, *_args):
        return None


class _TextEncoderInfo:
    @contextmanager
    def model_on_device(self):
        yield ({}, _TextEncoder())


def _identifier(key: str, model_type: ModelType, base: BaseModelType = BaseModelType.Any) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _invocation() -> Krea2TextEncoderInvocation:
    encoder = _identifier("encoder", ModelType.Qwen3VLEncoder)
    lora = _identifier("lora", ModelType.LoRA, BaseModelType.Krea2)
    field = Qwen3VLEncoderField(
        tokenizer=encoder.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=[LoRAField(lora=lora, weight=0.5)],
    )
    return Krea2TextEncoderInvocation.model_construct(prompt="a prompt", qwen3_vl_encoder=field)


def _context(lora_model) -> SimpleNamespace:
    def load(identifier):
        if identifier.key == "lora":
            return SimpleNamespace(model=lora_model)
        if identifier.submodel_type is SubModelType.Tokenizer:
            return _TokenizerInfo()
        return _TextEncoderInfo()

    return SimpleNamespace(
        models=SimpleNamespace(load=load), util=SimpleNamespace(signal_progress=lambda _message: None)
    )


def test_encode_applies_qwen3_vl_lora_and_returns_selected_hidden_layers(monkeypatch) -> None:
    captured = {}

    def apply_patches(**kwargs):
        captured.update(kwargs)
        captured["patches"] = list(kwargs["patches"])
        return nullcontext()

    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.LayerPatcher.apply_smart_model_patches", apply_patches
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    embeds, mask = _invocation()._encode(_context(ModelPatchRaw(layers={})))

    assert embeds.shape == (1, 6, 12, 4)
    assert mask is None
    assert captured["prefix"] == KREA2_LORA_QWEN3VL_PREFIX
    assert captured["patches"][0][1] == 0.5


def test_encode_rejects_a_loaded_non_patch_lora(monkeypatch) -> None:
    def apply_patches(**kwargs):
        list(kwargs["patches"])
        return nullcontext()

    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.LayerPatcher.apply_smart_model_patches", apply_patches
    )

    with pytest.raises(TypeError, match="Expected ModelPatchRaw"):
        _invocation()._encode(_context(object()))
