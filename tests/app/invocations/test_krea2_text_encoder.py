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
    def __call__(self, _text, max_length=None, **_kwargs):
        seq_len = max_length if max_length is not None else 5
        attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
        attention_mask[:, : min(seq_len, 35)] = 1
        return SimpleNamespace(
            input_ids=torch.ones((1, seq_len), dtype=torch.long),
            attention_mask=attention_mask,
        )


class _TextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **_kwargs):
        hidden_states = tuple(torch.full((1, input_ids.shape[1], 4), float(index)) for index in range(36))
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

    assert embeds.shape == (1, 512, 12, 4)
    assert mask is not None
    assert mask.shape == (1, 512)
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


def test_encode_preserves_suffix_for_a_prompt_that_overflows_truncation(monkeypatch) -> None:
    # Regression: a prompt longer than the tokenizer budget must NOT lose the assistant-turn suffix. The
    # encoder tokenizes (prefix + prompt) with truncation and appends the suffix AFTER, so the final tokens
    # always end with the suffix template (building one string and truncating it would cut the suffix off).
    from invokeai.app.invocations.krea2_text_encoder import _KREA2_SUFFIX

    suffix_ids = [901, 902, 903, 904, 905]

    class _TruncatingTokenizer:
        def __call__(
            self,
            text,
            max_length=None,
            truncation=False,
            padding=None,
            add_special_tokens=True,
            return_tensors=None,
        ):
            if text == _KREA2_SUFFIX:
                ids = list(suffix_ids)
            else:
                # Body (prefix + prompt): one filler id per whitespace token, distinct from the suffix ids.
                ids = [1] * len(text.split())
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]  # right truncation, as the real tokenizer does
            valid_length = len(ids)
            if padding == "max_length" and max_length is not None and len(ids) < max_length:
                ids.extend([0] * (max_length - len(ids)))
            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            attention_mask[:, :valid_length] = 1
            return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    captured: dict = {}

    class _CapturingEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *, input_ids, attention_mask, **_kwargs):
            captured["input_ids"] = input_ids
            seq = input_ids.shape[1]
            hidden_states = tuple(torch.zeros((1, seq, 4)) for _ in range(36))
            return SimpleNamespace(hidden_states=hidden_states)

    encoder = _CapturingEncoder()

    class _CapturingEncoderInfo:
        @contextmanager
        def model_on_device(self):
            yield ({}, encoder)

    class _TruncatingTokenizerInfo:
        def __enter__(self):
            return _TruncatingTokenizer()

        def __exit__(self, *_args):
            return None

    def load(identifier):
        if identifier.submodel_type is SubModelType.Tokenizer:
            return _TruncatingTokenizerInfo()
        return _CapturingEncoderInfo()

    context = SimpleNamespace(
        models=SimpleNamespace(load=load), util=SimpleNamespace(signal_progress=lambda _message: None)
    )

    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    encoder_id = _identifier("encoder", ModelType.Qwen3VLEncoder)
    field = Qwen3VLEncoderField(
        tokenizer=encoder_id.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder_id.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=[],
    )
    long_prompt = " ".join(["word"] * 2000)  # far exceeds the ~541-token budget
    invocation = Krea2TextEncoderInvocation.model_construct(prompt=long_prompt, qwen3_vl_encoder=field)

    invocation._encode(context)

    final_ids = captured["input_ids"][0].tolist()
    # The suffix survives at the very end even though the body was truncated.
    assert final_ids[-len(suffix_ids) :] == suffix_ids
    # And the body really was truncated (total = reserved budget + suffix), proving append-after-truncate.
    assert len(final_ids) > len(suffix_ids)


def test_encode_uses_reference_fixed_length_layout_and_position_ids(monkeypatch) -> None:
    from invokeai.app.invocations.krea2_text_encoder import _KREA2_SUFFIX

    captured: dict = {}

    class _ReferenceLayoutTokenizer:
        def __call__(
            self,
            text,
            max_length=None,
            truncation=False,
            padding=None,
            add_special_tokens=True,
            return_tensors=None,
        ):
            if text == _KREA2_SUFFIX:
                input_ids = torch.tensor([[91, 92, 93, 94, 95]], dtype=torch.long)
                return SimpleNamespace(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

            assert truncation is True
            assert padding == "max_length"
            assert max_length is not None
            input_ids = torch.zeros((1, max_length), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            input_ids[:, :4] = torch.tensor([[11, 12, 13, 14]])
            attention_mask[:, :4] = 1
            return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

    class _ReferenceLayoutEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *, input_ids, attention_mask, position_ids, **_kwargs):
            captured["input_ids"] = input_ids
            captured["attention_mask"] = attention_mask
            captured["position_ids"] = position_ids
            seq_len = input_ids.shape[1]
            hidden_states = tuple(torch.zeros((1, seq_len, 4)) for _ in range(36))
            return SimpleNamespace(hidden_states=hidden_states)

    class _ReferenceLayoutTokenizerInfo:
        def __enter__(self):
            return _ReferenceLayoutTokenizer()

        def __exit__(self, *_args):
            return None

    encoder = _ReferenceLayoutEncoder()

    class _ReferenceLayoutEncoderInfo:
        @contextmanager
        def model_on_device(self):
            yield ({}, encoder)

    encoder_id = _identifier("encoder", ModelType.Qwen3VLEncoder)
    field = Qwen3VLEncoderField(
        tokenizer=encoder_id.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder_id.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=[],
    )
    invocation = Krea2TextEncoderInvocation.model_construct(prompt="short prompt", qwen3_vl_encoder=field)

    def load(identifier):
        if identifier.submodel_type is SubModelType.Tokenizer:
            return _ReferenceLayoutTokenizerInfo()
        return _ReferenceLayoutEncoderInfo()

    context = SimpleNamespace(
        models=SimpleNamespace(load=load), util=SimpleNamespace(signal_progress=lambda _message: None)
    )

    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.krea2_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    embeds, mask = invocation._encode(context)

    assert embeds.shape == (1, 512, 12, 4)
    assert mask is not None
    assert mask.shape == (1, 512)
    assert captured["input_ids"].shape == (1, 546)
    assert captured["attention_mask"].dtype == torch.bool
    assert captured["position_ids"].shape == (3, 1, 546)
    assert captured["position_ids"][0, 0, -5:].tolist() == [4, 5, 6, 7, 8]
