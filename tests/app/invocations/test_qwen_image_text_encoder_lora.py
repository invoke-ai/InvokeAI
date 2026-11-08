"""LoRA layers aimed at the Qwen2.5-VL encoder reach it on every load path of the Qwen-Image text encoder node."""

import gc
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, QwenVLEncoderField
from invokeai.app.invocations.text_encoder.qwen_image_text_encoder import (
    _GENERATE_DROP_IDX,
    QwenImageTextEncoderInvocation,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    lora_model_from_qwen_image_state_dict,
)
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from invokeai.backend.util.devices import TorchDevice

HIDDEN = 8
RANK = 4
_PROJ_PATH = "model.language_model.layers.0.mlp.down_proj"
SEQ_LEN = _GENERATE_DROP_IDX + 3


class _InnerModel(torch.nn.Module):
    """The encoder's inner model, which the node runs: its forward returns the Linear's output as the last hidden state."""

    def __init__(self, fail: bool) -> None:
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.language_model.layers[0].mlp = torch.nn.Module()
        self.language_model.layers[0].mlp.down_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.fail = fail
        self.patches_during_forward: int | None = None

    def forward(self, input_ids, attention_mask, **_):
        proj = self.language_model.layers[0].mlp.down_proj
        self.patches_during_forward = proj.get_num_patches() if hasattr(proj, "get_num_patches") else None
        if self.fail:
            raise RuntimeError("forward failed (simulated)")
        x = torch.linspace(-1, 1, input_ids.numel() * HIDDEN).reshape(*input_ids.shape, HIDDEN)
        return SimpleNamespace(last_hidden_state=proj(x))


class _Encoder(torch.nn.Module):
    """One Linear the LoRA targets, at the path of a real Qwen2.5-VL Linear, which is what a converted LoRA can address."""

    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.model = _InnerModel(fail)

    @property
    def proj(self) -> torch.nn.Module:
        return self.model.language_model.layers[0].mlp.down_proj

    @property
    def patches_during_forward(self) -> int | None:
        return self.model.patches_during_forward


def _encoder_module(fail: bool = False) -> _Encoder:
    torch.manual_seed(0)
    return _Encoder(fail)


def _identifier(key: str, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash="h", name=key, base=BaseModelType.QwenImage, type=model_type)


def _lora_weights() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(RANK, HIDDEN, generator=generator), torch.randn(HIDDEN, RANK, generator=generator)


_PIN = nullcontext()  # what `model_in_ram()` hands back: the cache pin the patcher must hold while patching


def _lora_info() -> SimpleNamespace:
    down, up = _lora_weights()
    patch_raw = lora_model_from_qwen_image_state_dict(
        {
            f"text_encoder.{_PROJ_PATH}.lora_A.weight": down,
            f"text_encoder.{_PROJ_PATH}.lora_B.weight": up,
        }
    )
    return SimpleNamespace(model=patch_raw, model_in_ram=lambda: _PIN)


def _node(loras: list[LoRAField], quantization: str = "none") -> QwenImageTextEncoderInvocation:
    encoder = _identifier("enc", ModelType.QwenVLEncoder)
    return QwenImageTextEncoderInvocation.model_construct(
        prompt="a cat",
        reference_images=[],
        qwen_vl_encoder=QwenVLEncoderField(tokenizer=encoder, text_encoder=encoder, loras=loras),
        quantization=quantization,
    )


def _lora_field(weight: float) -> LoRAField:
    return LoRAField(lora=_identifier("lora", ModelType.LoRA), weight=weight)


def _context(
    model_root: Path, text_encoder_info: object | None = None, model_format: ModelFormat = ModelFormat.Diffusers
):
    """A context whose model loads serve the LoRA, the tokenizer (opaque to the node) and the cached encoder."""
    context = MagicMock()
    context.models.get_absolute_path.return_value = model_root
    context.models.get_config.return_value = SimpleNamespace(format=model_format)

    def load(identifier):
        if identifier.type == ModelType.LoRA:
            return _lora_info()
        if identifier.submodel_type is None and text_encoder_info is not None:
            return text_encoder_info
        return SimpleNamespace(model=MagicMock())

    context.models.load.side_effect = load
    return context


def _cached_info(encoder: torch.nn.Module, events: list, cached_weights: object = None) -> SimpleNamespace:
    @contextmanager
    def model_on_device(working_mem_bytes=None):
        events.append("lock")
        try:
            yield cached_weights, encoder
        finally:
            events.append(("release", getattr(encoder.proj, "get_num_patches", lambda: None)()))

    return SimpleNamespace(model=encoder, compute_device=torch.device("cpu"), model_on_device=model_on_device)


def _processor() -> MagicMock:
    model_inputs = MagicMock()
    model_inputs.input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    model_inputs.attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)
    model_inputs.to.return_value = model_inputs
    return MagicMock(return_value=model_inputs)


def _encode(node: QwenImageTextEncoderInvocation, context: MagicMock) -> torch.Tensor:
    with patch.object(QwenImageTextEncoderInvocation, "_build_processor", return_value=_processor()):
        prompt_embeds, _ = node._encode(context, images=[])
    return prompt_embeds


def _expected_shift(weight: float) -> torch.Tensor:
    """What a LoRA adds to the encoder output: `x @ (up @ down).T * weight` (no alpha, so scale 1), in bf16."""
    down, up = _lora_weights()
    x = torch.linspace(-1, 1, SEQ_LEN * HIDDEN).reshape(1, SEQ_LEN, HIDDEN)
    return (x @ (up @ down).T * weight)[:, _GENERATE_DROP_IDX:].to(torch.bfloat16).float()


@pytest.fixture(autouse=True)
def _float32_patch_dtype(monkeypatch: pytest.MonkeyPatch):
    """The node patches in bf16; CPU bf16 matmul faults on part of the CI fleet (tests/AGENTS.md), so run float32."""
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda device: torch.float32)


@pytest.fixture
def _tiny_encoder_class(monkeypatch: pytest.MonkeyPatch):
    """The cached loader checks its model's type; the test encoder stands in for the real class."""
    import transformers

    monkeypatch.setattr(transformers, "Qwen2_5_VLForConditionalGeneration", _Encoder, raising=False)


class _RecordedPatching(Exception):
    pass


@pytest.mark.parametrize(
    ("packed", "model_format", "sidecar"),
    [
        (True, ModelFormat.Checkpoint, True),
        (False, ModelFormat.Checkpoint, False),
        (False, ModelFormat.GGUFQuantized, True),
    ],
    ids=["nvfp4", "dense", "gguf"],
)
@pytest.mark.usefixtures("_tiny_encoder_class")
def test_cached_path_hands_prefix_weights_and_cached_weights_to_the_patcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, packed: bool, model_format: ModelFormat, sidecar: bool
) -> None:
    encoder = _encoder_module()
    if packed:
        encoder.packed = NVFP4Linear(
            torch.zeros(128, 32, dtype=torch.uint8), torch.ones(128, 4).to(torch.float8_e4m3fn), torch.tensor(1.0)
        )
    cached_weights = {f"{_PROJ_PATH}.weight": torch.zeros(1)}
    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)
        seen["patches"] = list(kwargs["patches"])
        raise _RecordedPatching

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", record)
    events: list = []
    context = _context(tmp_path, _cached_info(encoder, events, cached_weights), model_format)

    with pytest.raises(_RecordedPatching):
        _node([_lora_field(0.7)])._load_cached_encoder(context)

    assert seen["prefix"] == "lora_qwen_vl-"
    assert seen["cached_weights"] is cached_weights
    assert seen["force_sidecar_patching"] is sidecar
    assert [spec[1] for spec in seen["patches"]] == [0.7]
    assert seen["patches"][0][2] is _PIN
    # The failed patching must not leave the model locked.
    assert events[-1][0] == "release"


@pytest.mark.usefixtures("_tiny_encoder_class")
def test_cached_path_applies_the_lora_to_the_encoder_output_and_removes_it_afterwards(tmp_path: Path) -> None:
    """The encoder output moves by the LoRA contribution, the layer carries no patch once the node is done, and the
    patch is gone before the model lock is released."""
    encoder = _encoder_module()
    apply_custom_layers_to_model(encoder)  # as the model cache does on `put()`
    assert isinstance(encoder.proj, CustomLinear)
    events: list = []
    context = _context(tmp_path, _cached_info(encoder, events))

    baseline = _encode(_node([]), context)
    patched = _encode(_node([_lora_field(0.5)]), context)
    patches_during_patched_forward = encoder.patches_during_forward
    again = _encode(_node([]), context)

    torch.testing.assert_close((patched - baseline).float(), _expected_shift(0.5), atol=0.1, rtol=0.05)
    assert not torch.equal(patched, baseline)
    assert torch.equal(again, baseline)
    assert patches_during_patched_forward == 1
    assert events == ["lock", ("release", 0)] * 3


@pytest.mark.usefixtures("_tiny_encoder_class")
def test_cached_path_undoes_patches_and_releases_the_lock_when_the_forward_raises(tmp_path: Path) -> None:
    encoder = _encoder_module(fail=True)
    apply_custom_layers_to_model(encoder)
    events: list = []
    context = _context(tmp_path, _cached_info(encoder, events))

    with pytest.raises(RuntimeError, match="simulated"):
        _encode(_node([_lora_field(1.0)]), context)

    assert encoder.patches_during_forward == 1
    assert events == ["lock", ("release", 0)]


def _quantized_context(tmp_path: Path) -> MagicMock:
    model_root = tmp_path / "qwen-vl"
    (model_root / "text_encoder").mkdir(parents=True, exist_ok=True)
    Qwen2_5_VLConfig().save_pretrained(model_root / "text_encoder")
    return _context(model_root)


@contextmanager
def _quantized_load(handoff: list):
    """`from_pretrained` serves the test encoder (a one-shot handoff, so no mock keeps it alive) on the CPU."""
    with (
        patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=lambda *a, **k: handoff.pop()),
        patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
    ):
        yield


def test_quantized_path_without_loras_leaves_the_layers_unwrapped(tmp_path: Path) -> None:
    encoder = _encoder_module()
    with _quantized_load([encoder]):
        loaded, _, cleanup = _node([], "int8")._load_quantized_encoder(_quantized_context(tmp_path))

    assert loaded is encoder
    assert type(encoder.proj) is torch.nn.Linear
    cleanup()


@pytest.mark.parametrize("failure", [None, "forward", "patching"], ids=["encoded", "forward_raises", "patching_raises"])
def test_quantized_path_frees_the_patched_encoder_before_the_cache_is_emptied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str | None
) -> None:
    """The patcher's generator frames hold the model, so the patches must be closed before cleanup empties the cache
    (see `_encode`): otherwise the encoder's VRAM stays reserved by torch. A failure while patching releases it too."""
    encoder = _encoder_module(failure == "forward")
    encoder_ref = weakref.ref(encoder)
    freed_at_empty_cache: list[bool] = []
    handoff = [encoder]
    del encoder
    if failure == "patching":

        def fail(**_kwargs):
            raise RuntimeError("patching failed (simulated)")

        monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", fail)

    def empty_cache() -> None:
        gc.collect()
        freed_at_empty_cache.append(encoder_ref() is None)

    with _quantized_load(handoff), patch.object(TorchDevice, "empty_cache", side_effect=empty_cache):
        node = _node([_lora_field(0.5)], "int8")
        if failure:
            with pytest.raises(RuntimeError, match="simulated"):
                _encode(node, _quantized_context(tmp_path))
        else:
            _encode(node, _quantized_context(tmp_path))

    assert freed_at_empty_cache == [True]


def test_an_oom_while_patching_is_not_reported_as_the_encoder_not_fitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def oom(**_kwargs):
        raise torch.OutOfMemoryError("oom while moving the LoRA")

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", oom)

    with _quantized_load([_encoder_module()]), pytest.raises(torch.OutOfMemoryError) as excinfo:
        _node([_lora_field(0.5)], "int8")._load_quantized_encoder(_quantized_context(tmp_path))

    assert str(excinfo.value) == "oom while moving the LoRA"


def test_quantized_path_always_patches_as_a_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """On the CPU the patcher picks a sidecar anyway, so only the flag itself shows that a quantized layer is never
    patched in place on a device where it would be."""
    seen: dict = {}

    def record(**kwargs):
        seen.update(kwargs)
        raise _RecordedPatching

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", record)

    with _quantized_load([_encoder_module()]), pytest.raises(_RecordedPatching):
        _node([_lora_field(0.5)], "int8")._load_quantized_encoder(_quantized_context(tmp_path))

    assert seen["prefix"] == "lora_qwen_vl-"
    assert seen["force_sidecar_patching"] is True
    assert seen["cached_weights"] is None


def test_quantized_path_changes_the_output_by_the_lora_contribution(tmp_path: Path) -> None:
    results = []
    for loras in ([], [_lora_field(0.5)]):
        with _quantized_load([_encoder_module()]):
            results.append(_encode(_node(loras, "int8"), _quantized_context(tmp_path)))

    torch.testing.assert_close((results[1] - results[0]).float(), _expected_shift(0.5), atol=0.1, rtol=0.05)


@pytest.mark.usefixtures("_tiny_encoder_class")
def test_quantized_single_file_fallback_patches_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A single-file checkpoint is served by the cached loader, which already applies the LoRA."""
    single_file = tmp_path / "encoder.safetensors"
    single_file.write_bytes(b"")
    encoder = _encoder_module()
    apply_custom_layers_to_model(encoder)
    calls: list[str] = []
    original = LayerPatcher.apply_smart_model_patches

    def counting(**kwargs):
        calls.append(kwargs["prefix"])
        return original(**kwargs)

    monkeypatch.setattr(LayerPatcher, "apply_smart_model_patches", counting)
    context = _context(single_file, _cached_info(encoder, []))

    _encode(_node([_lora_field(1.0)], "int8"), context)

    assert calls == ["lora_qwen_vl-"]
