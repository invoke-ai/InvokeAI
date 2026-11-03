from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import TensorField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, Qwen3VLEncoderField
from invokeai.app.invocations.text_encoder.krea2_text_encoder import Krea2TextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_QWEN3VL_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# These fakes stand in for a dense safetensors encoder, so LoRA is applied directly. The sidecar
# decision for a quantized one is covered in test_text_encoders_with_packed_layers.py.
_ENCODER_CONFIG = SimpleNamespace(format=ModelFormat.Checkpoint)


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
    # `model` and `compute_device` are read before the lock, to size the dequant transient the node
    # reserves for a packed encoder.
    model = _TextEncoder()
    compute_device = torch.device("cpu")

    @contextmanager
    def model_on_device(self, working_mem_bytes=None):
        yield ({}, self.model)


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


class _LoRAInfo:
    """Stands in for LoadedModel: the encoder pins the LoRA's cache record for the patch's lifetime."""

    def __init__(self, model) -> None:
        self.model = model
        self.pin_depth = 0
        self.pinned_at_least_once = False

    @contextmanager
    def model_in_ram(self):
        self.pin_depth += 1
        self.pinned_at_least_once = True
        try:
            yield self.model
        finally:
            self.pin_depth -= 1


def _context(lora_model, lora_infos: list | None = None) -> SimpleNamespace:
    def load(identifier):
        if identifier.key == "lora":
            info = _LoRAInfo(lora_model)
            if lora_infos is not None:
                lora_infos.append(info)
            return info
        if identifier.submodel_type is SubModelType.Tokenizer:
            return _TokenizerInfo()
        return _TextEncoderInfo()

    return SimpleNamespace(
        models=SimpleNamespace(load=load, get_config=lambda _identifier: _ENCODER_CONFIG),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )


def test_encode_applies_qwen3_vl_lora_and_returns_selected_hidden_layers(monkeypatch) -> None:
    captured = {}

    def apply_patches(**kwargs):
        captured.update(kwargs)
        captured["patches"] = list(kwargs["patches"])
        return nullcontext()

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.LayerPatcher.apply_smart_model_patches", apply_patches
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    lora_infos: list[_LoRAInfo] = []
    embeds, mask = _invocation()._encode(_context(ModelPatchRaw(layers={}), lora_infos))

    assert embeds.shape == (1, 512, 12, 4)
    assert mask is not None
    assert mask.shape == (1, 512)
    assert captured["prefix"] == KREA2_LORA_QWEN3VL_PREFIX
    assert captured["patches"][0][1] == 0.5

    # The patch spec must carry a cache pin so LayerPatcher can hold the LoRA's record in RAM for the
    # lifetime of the patch. The patcher itself is stubbed out here, so the pin arrives un-entered.
    assert len(captured["patches"][0]) == 3
    assert lora_infos[0].pin_depth == 0
    with captured["patches"][0][2]:
        assert lora_infos[0].pin_depth == 1
    assert lora_infos[0].pin_depth == 0


def test_encode_rejects_a_loaded_non_patch_lora(monkeypatch) -> None:
    def apply_patches(**kwargs):
        list(kwargs["patches"])
        return nullcontext()

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.LayerPatcher.apply_smart_model_patches", apply_patches
    )

    with pytest.raises(TypeError, match="Expected ModelPatchRaw"):
        _invocation()._encode(_context(object()))


def test_encode_preserves_suffix_for_a_prompt_that_overflows_truncation(monkeypatch) -> None:
    # Regression: a prompt longer than the tokenizer budget must NOT lose the assistant-turn suffix. The
    # encoder tokenizes (prefix + prompt) with truncation and appends the suffix AFTER, so the final tokens
    # always end with the suffix template (building one string and truncating it would cut the suffix off).
    from invokeai.backend.krea2.text_encoding import KREA2_SUFFIX as _KREA2_SUFFIX

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
        model = encoder
        compute_device = torch.device("cpu")

        @contextmanager
        def model_on_device(self, working_mem_bytes=None):
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
        models=SimpleNamespace(load=load, get_config=lambda _identifier: _ENCODER_CONFIG),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
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
    from invokeai.backend.krea2.text_encoding import KREA2_SUFFIX as _KREA2_SUFFIX

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
        model = encoder
        compute_device = torch.device("cpu")

        @contextmanager
        def model_on_device(self, working_mem_bytes=None):
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
        models=SimpleNamespace(load=load, get_config=lambda _identifier: _ENCODER_CONFIG),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.LayerPatcher.apply_smart_model_patches",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
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


def test_visual_tower_lora_layers_are_skipped_without_blaming_the_adapter(monkeypatch) -> None:
    """The loader drops the Qwen3-VL visual tower, so an adapter carrying layers for it finds no
    module. The converter really does emit these keys (see `test_krea2_lora_conversion_utils`), and
    the 4B's vision tower has 27 blocks, so left to the patcher this is ~108 "Failed to find module
    for LoRA layer key" lines per generation -- reading as an adapter failure for layers that never
    affected an image.

    Asserted on the pattern handed to the patcher, since the suppression is the patcher's to apply.
    """
    import re

    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.krea2_text_encoder.LayerPatcher.apply_smart_model_patches",
        capture,
    )

    _invocation()._encode(_context(ModelPatchRaw(layers={})))

    pattern = captured["suppress_warning_layers"]
    assert isinstance(pattern, re.Pattern)
    assert pattern.search(f"{KREA2_LORA_QWEN3VL_PREFIX}visual.blocks.0.attn.qkv")
    # The language tower is what the adapter is for; a miss there is a real problem and must still
    # be reported.
    assert not pattern.search(f"{KREA2_LORA_QWEN3VL_PREFIX}layers.0.self_attn.q_proj")


def test_invoke_preserves_the_regional_mask_on_its_conditioning_output(monkeypatch) -> None:
    regional_mask = TensorField(tensor_name="regional-mask")
    invocation = _invocation().model_copy(update={"mask": regional_mask})
    monkeypatch.setattr(
        invocation,
        "_encode",
        lambda _context: (torch.zeros(1, 2, 12, 4), torch.ones(1, 2, dtype=torch.bool)),
    )
    context = SimpleNamespace(conditioning=SimpleNamespace(save=lambda _data: "conditioning-name"))

    output = invocation.invoke(context)

    assert output.conditioning.conditioning_name == "conditioning-name"
    assert output.conditioning.mask == regional_mask


# --- reference images (edit mode) ------------------------------------------------------------------

_INVOCATION_MODULE = "invokeai.app.invocations.text_encoder.krea2_text_encoder"


class _RecordingInfo:
    """A LoadedModel stand-in that logs when its model is locked and released."""

    def __init__(self, name: str, model, events: list[str]) -> None:
        self.name = name
        self.model = model
        self.compute_device = torch.device("cpu")
        self._events = events
        self.working_mem_bytes: list[int | None] = []

    @contextmanager
    def model_on_device(self, working_mem_bytes=None):
        self.working_mem_bytes.append(working_mem_bytes)
        self._events.append(f"lock {self.name}")
        try:
            yield ({}, self.model)
        finally:
            self._events.append(f"unlock {self.name}")

    def __enter__(self):
        return self.model

    def __exit__(self, *_args):
        return None


# "strip" keeps its 6000 px width and is held at 28 px tall: an aspect ratio over the Qwen3-VL processor's 200.
_IMAGE_SIZES = {"big": (800, 600), "small": (64, 48), "strip": (6000, 20)}


@pytest.fixture(scope="module")
def processing() -> dict:
    """The bundled Qwen3-VL tokenizer and processor, the small CPU-side models an image encode loads first."""
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import build_qwen3_vl_processor, load_bundled_qwen3_vl_tokenizer

    tokenizer = load_bundled_qwen3_vl_tokenizer()
    return {SubModelType.Tokenizer: tokenizer, SubModelType.Processor: build_qwen3_vl_processor(tokenizer)}


def _edit_context(
    events: list[str], loaded: list, infos: dict, *, refuse_tower: bool = False, processing: dict | None = None
) -> SimpleNamespace:
    """Records every model loaded, in order. `processing` swaps in real models by submodel type."""
    from PIL import Image

    def load(identifier):
        loaded.append(identifier)
        if refuse_tower and identifier.submodel_type is SubModelType.VisionEncoder:
            raise ValueError("The GGUF Qwen3-VL encoder 'encoder' has no vision tower")
        name = identifier.submodel_type.value
        fakes = {SubModelType.TextEncoder: _TextEncoder, SubModelType.Tokenizer: _Tokenizer}
        if processing is not None and identifier.submodel_type in processing:
            model = processing[identifier.submodel_type]
        else:
            model = fakes.get(identifier.submodel_type, object)()
        infos[name] = _RecordingInfo(name, model, events)
        return infos[name]

    return SimpleNamespace(
        models=SimpleNamespace(load=load, get_config=lambda _identifier: _ENCODER_CONFIG),
        util=SimpleNamespace(signal_progress=lambda _message: None),
        images=SimpleNamespace(get_pil=lambda name: Image.new("RGB", _IMAGE_SIZES.get(name, (64, 48)))),
    )


def _edit_invocation(*image_names: str) -> Krea2TextEncoderInvocation:
    from invokeai.app.invocations.fields import ImageField

    encoder = _identifier("encoder", ModelType.Qwen3VLEncoder)
    field = Qwen3VLEncoderField(
        tokenizer=encoder.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
        text_encoder=encoder.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        loras=[],
    )
    return Krea2TextEncoderInvocation(
        prompt="make it red",
        qwen3_vl_encoder=field,
        reference_images=[ImageField(image_name=name) for name in image_names],
    )


@pytest.fixture
def _no_lora_patching(monkeypatch):
    monkeypatch.setattr(f"{_INVOCATION_MODULE}.LayerPatcher.apply_smart_model_patches", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(f"{_INVOCATION_MODULE}.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: torch.float32)


def test_a_text_only_prompt_loads_neither_the_tower_nor_the_processor(_no_lora_patching) -> None:
    loaded: list = []
    events: list[str] = []
    infos: dict = {}

    embeds, _ = _edit_invocation()._encode(_edit_context(events, loaded, infos))

    assert {identifier.submodel_type for identifier in loaded} == {SubModelType.Tokenizer, SubModelType.TextEncoder}
    assert embeds.shape == (1, 512, 12, 4)
    assert infos["text_encoder"].working_mem_bytes == [0]


def test_reference_images_lend_the_tower_to_the_locked_encoder(monkeypatch, _no_lora_patching) -> None:
    """The tower and processor come from the encoder's own model, the encode reserves for the longer
    sequence, and the tower is locked inside the encoder's lock and released before it: neither cache
    entry may move while the tower is part of the encoder's module tree."""
    from invokeai.backend.krea2.text_encoding import working_memory_bytes

    loaded: list = []
    events: list[str] = []
    infos: dict = {}
    captured: dict = {}

    def tokenize(prompt, images, processor, tokenizer):
        captured.update(prompt=prompt, images=images, processor=processor)
        return {"input_ids": torch.zeros(1, 300, dtype=torch.long)}

    def encode(inputs, text_encoder, vision_tower):
        events.append("encode")
        captured.update(text_encoder=text_encoder, vision_tower=vision_tower)
        return torch.zeros(1, 266, 12, 4), torch.ones(1, 266, dtype=torch.bool)

    monkeypatch.setattr(f"{_INVOCATION_MODULE}.tokenize_krea2_prompt_with_images", tokenize)
    monkeypatch.setattr(f"{_INVOCATION_MODULE}.encode_krea2_prompt_with_images", encode)

    embeds, mask = _edit_invocation("big", "small")._encode(_edit_context(events, loaded, infos))

    by_type = {identifier.submodel_type: identifier for identifier in loaded}
    assert by_type[SubModelType.VisionEncoder].key == "encoder"
    assert by_type[SubModelType.Processor].key == "encoder"
    assert captured["prompt"] == "make it red"
    # Each reference went through the 384^2 rule on the way: 800x600 shrinks, 64x48 stays.
    assert [tuple(image.shape) for image in captured["images"]] == [(3, 333, 443), (3, 48, 64)]
    assert captured["processor"] is infos["processor"].model
    assert captured["text_encoder"] is infos["text_encoder"].model
    assert captured["vision_tower"] is infos["vision_encoder"].model
    assert infos["text_encoder"].working_mem_bytes == [working_memory_bytes(300)]
    assert events == [
        "lock text_encoder",
        "lock vision_encoder",
        "encode",
        "unlock vision_encoder",
        "unlock text_encoder",
    ]
    assert embeds.shape == (1, 266, 12, 4)
    assert mask is not None and bool(mask.all())


def test_the_tower_is_locked_with_the_encoders_reservation(monkeypatch, _no_lora_patching) -> None:
    """The cache keeps a lock's reservation free beyond everything resident, the locked encoder included. The
    tower is locked second: with the default reservation its weights could take the room reserved for the
    encoder's forward. A dequant transient is part of that room, so it must reach the tower's lock as well."""
    from invokeai.backend.krea2.text_encoding import working_memory_bytes

    infos: dict = {}
    dequant_bytes = 7 * 2**20
    monkeypatch.setattr(f"{_INVOCATION_MODULE}.peak_dequant_transient_bytes", lambda _model, _dtype: dequant_bytes)
    monkeypatch.setattr(
        f"{_INVOCATION_MODULE}.tokenize_krea2_prompt_with_images",
        lambda *_args: {"input_ids": torch.zeros(1, 300, dtype=torch.long)},
    )
    monkeypatch.setattr(
        f"{_INVOCATION_MODULE}.encode_krea2_prompt_with_images",
        lambda *_args: (torch.zeros(1, 266, 12, 4), torch.ones(1, 266, dtype=torch.bool)),
    )

    _edit_invocation("small")._encode(_edit_context([], [], infos))

    reservation = dequant_bytes + working_memory_bytes(300)
    assert infos["text_encoder"].working_mem_bytes == [reservation]
    assert infos["vision_encoder"].working_mem_bytes == [reservation]


def test_a_reference_the_processor_refuses_fails_before_either_tower_is_read(_no_lora_patching, processing) -> None:
    """Only the processor and tokenizer are needed to find out: the ~8 GB language tower, and the vision
    tower, are not read for a prompt that cannot run."""
    loaded: list = []

    with pytest.raises(ValueError, match="reference image 2 cannot be read by the Qwen3-VL vision tower"):
        _edit_invocation("small", "strip")._encode(_edit_context([], loaded, {}, processing=processing))

    assert [identifier.submodel_type for identifier in loaded] == [SubModelType.Tokenizer, SubModelType.Processor]


def test_an_encoder_without_a_tower_fails_before_the_encoder_is_read(_no_lora_patching, processing) -> None:
    loaded: list = []

    with pytest.raises(ValueError, match="has no vision tower"):
        _edit_invocation("small")._encode(_edit_context([], loaded, {}, refuse_tower=True, processing=processing))

    assert SubModelType.TextEncoder not in {identifier.submodel_type for identifier in loaded}


def test_more_than_three_reference_images_are_refused() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="reference_images"):
        _edit_invocation("a", "b", "c", "d")


@pytest.mark.parametrize("image_names", [(), ("big", "small")], ids=["text_only", "two_references"])
def test_the_conditioning_records_how_many_references_it_read(monkeypatch, image_names) -> None:
    saved: dict = {}
    invocation = _edit_invocation(*image_names)
    monkeypatch.setattr(
        invocation, "_encode", lambda _context: (torch.zeros(1, 2, 12, 4), torch.ones(1, 2, dtype=torch.bool))
    )
    context = SimpleNamespace(conditioning=SimpleNamespace(save=lambda data: saved.setdefault("data", data) and "c"))

    invocation.invoke(context)

    assert saved["data"].conditionings[0].num_reference_images == len(image_names)
