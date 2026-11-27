"""The Qwen-Image text encoder's single-file loader on ComfyUI's `qwen_2.5_vl_7b_fp8_scaled` layout.

The real `Qwen2_5_VLForConditionalGeneration` is built at toy width from the vendored config (two language-model
layers, two vision blocks), and the file is written in the legacy key layout with that build's `scale_weight` /
`scale_input` spellings: every Linear (language model, vision tower, untied LM head) is float8 beside its scale, the
token table is scaled too, and the patch-embedding convolution and norms are dense. The LM head stays in the file. On a
device that can hold float8 the Linears stay float8 with the file's own scale on their module; the token table is
folded; elsewhere everything is folded. Expected values come from the codes and scales written, run through a second,
plain model. Forwards run in float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from invokeai.backend.model_manager.configs.qwen_vl_encoder import QwenVLEncoder_Checkpoint_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules import custom_linear
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import qwen_image
from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenVLEncoderCheckpointLoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.block_scale_tiles import block_scale_grid
from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import load_bundled_qwen2_5_vl_config_dict
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.quantized_payloads import comfy_quant_marker

FP8 = torch.float8_e4m3fn
# 128x128: the smallest weight an MXFP8 scale grid (tiles of 128 rows by 4 blocks of 32) can describe.
Q_PROJ = "model.language_model.layers.0.self_attn.q_proj"
MX_MODULE = Q_PROJ


def _toy_config_dict() -> dict:
    config = load_bundled_qwen2_5_vl_config_dict()
    config.update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_window_layers=2,
        rope_scaling={"type": "mrope", "mrope_section": [4, 6, 6]},
    )
    config["vision_config"].update(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128, window_size=28
    )
    config["vision_config"]["fullatt_block_indexes"] = [1]
    # The vendored config says bfloat16, the sub-configs included.
    config["torch_dtype"] = config["vision_config"]["torch_dtype"] = "float32"
    return config


def _new_model() -> Qwen2_5_VLForConditionalGeneration:
    config = Qwen2_5_VLConfig.from_dict(_toy_config_dict())
    return Qwen2_5_VLForConditionalGeneration(config).eval()


def _legacy(name: str) -> str:
    """transformers' layout back to the file's: `model.language_model.X` -> `model.X`, `model.visual.X` -> `visual.X`."""
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    return name.removeprefix("model.") if name.startswith("model.visual.") else name


class _Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        # What the file decodes to, keyed by state-dict key of the plain model.
        self.dense: dict[str, torch.Tensor] = {}
        # Module path -> the scale written (per-row or scalar), for the layers that carry a plain one.
        self.scales: dict[str, torch.Tensor] = {}
        self.linear_modules: list[str] = []
        self.linear_elements = 0
        self.other_elements = 0
        self.scale_bytes = 0


def _write(
    tmp_path: Path,
    *,
    scaled: bool = True,
    evidence: str = "none",
    mx_module: str | None = None,
    extra: dict[str, torch.Tensor] | None = None,
) -> _Checkpoint:
    """`evidence` is how the file names the layer `mx_module` (or, without one, every scaled layer): per-layer
    `comfy_quant` markers, the safetensors header, or nothing."""
    torch.manual_seed(4)
    reference = _new_model()
    checkpoint = _Checkpoint(tmp_path / "qwen_2.5_vl_7b_fp8_scaled.safetensors")
    tensors: dict[str, torch.Tensor] = {"scaled_fp8": torch.empty(0, dtype=FP8)}
    hints: dict[str, dict] = {}
    count = 0
    for key, param in reference.state_dict().items():
        module_path, _, leaf = key.rpartition(".")
        module = reference.get_submodule(module_path)
        legacy_key, legacy_module = _legacy(key), _legacy(module_path)
        is_linear = leaf == "weight" and isinstance(module, torch.nn.Linear)
        # The file carries the LM head, but the loaded encoder does not, so no expectation about the model counts it.
        loaded = module_path != "lm_head"
        is_table = leaf == "weight" and module_path == "model.language_model.embed_tokens"
        if not (is_linear or is_table):
            value = torch.rand(param.shape) + 0.5 if "norm" in key else torch.randn(param.shape) * 0.3
            tensors[legacy_key] = value
            checkpoint.dense[key] = value
            checkpoint.other_elements += param.numel()
            continue
        codes = torch.randint(-12, 13, tuple(param.shape)).float()
        tensors[legacy_key] = codes.to(FP8)
        if not scaled:
            checkpoint.dense[key] = codes
        elif module_path == mx_module:
            exponents = torch.randint(125, 130, (param.shape[0], param.shape[1] // 32))
            tensors[f"{legacy_module}.weight_scale"] = block_scale_grid(exponents.to(torch.uint8))
            checkpoint.dense[key] = codes * torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=1)
            checkpoint.scale_bytes += 4 * exponents.numel()
            hints[legacy_module] = {"format": "mxfp8"}
        else:
            per_row = count % 2 == 0
            count += 1
            scale = torch.linspace(0.25, 1.5, param.shape[0]) if per_row else torch.tensor(0.5)
            tensors[f"{legacy_module}.scale_weight"] = scale
            # The placeholder this build writes beside every scale; 1.0 is not a calibration.
            tensors[f"{legacy_module}.scale_input"] = torch.tensor(1.0)
            checkpoint.dense[key] = codes * (scale[:, None] if per_row else scale)
            if loaded:
                checkpoint.scales[module_path] = scale
                checkpoint.scale_bytes += 4 * scale.numel()
            hints[legacy_module] = {"format": "float8_e4m3fn"}
        if not loaded:
            continue
        if is_linear:
            checkpoint.linear_modules.append(module_path)
            checkpoint.linear_elements += param.numel()
        else:
            checkpoint.other_elements += param.numel()
    tensors.update(extra or {})
    if mx_module is not None:
        hints = {_legacy(mx_module): {"format": "mxfp8"}}
    metadata = None
    if evidence == "marker":
        for legacy_module, marker in hints.items():
            tensors[f"{legacy_module}.comfy_quant"] = comfy_quant_marker(marker)
    elif evidence == "header":
        metadata = {"_quantization_metadata": json.dumps({"layers": hints})}
    save_file(tensors, checkpoint.path, metadata=metadata)
    return checkpoint


def _decide(monkeypatch: pytest.MonkeyPatch, *, device: str, matmul: bool = False, storage: bool = False) -> None:
    """The session runs on `device`; only a CUDA device passes the gates that are switched on."""
    monkeypatch.setattr(qwen_image, "load_bundled_qwen2_5_vl_config_dict", _toy_config_dict)
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_torch_device", lambda: torch.device(device))
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda d: matmul and d.type == "cuda")
    monkeypatch.setattr(
        load_default, "_device_supports_fp8_storage", lambda d, _logger=None: storage and d.type == "cuda"
    )


def _loader() -> QwenVLEncoderCheckpointLoader:
    loader = object.__new__(QwenVLEncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    # What `ModelLoader.__init__` reads, on the device `_decide` set.
    loader._torch_device, loader._logger = qwen_image.TorchDevice.choose_torch_device(), MagicMock()
    return loader


def _load(checkpoint: _Checkpoint, loader: QwenVLEncoderCheckpointLoader | None = None, config=None):
    if config is None:
        config = QwenVLEncoder_Checkpoint_Config.model_construct(path=str(checkpoint.path))
    return (loader or _loader())._load_text_encoder_from_singlefile(config)


def _plain_reference(dense: dict[str, torch.Tensor]) -> Qwen2_5_VLForConditionalGeneration:
    model = _new_model()
    model.load_state_dict(dense)
    return model


IDS = torch.tensor([[1, 5, 7, 2, 30]])
GRID = torch.tensor([[1, 4, 4]])
PIXELS = torch.randn(16, 3 * 2 * 14 * 14, generator=torch.Generator().manual_seed(9))


def _run(model: Qwen2_5_VLForConditionalGeneration) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return model.model(input_ids=IDS).last_hidden_state, model.model.visual(PIXELS, grid_thw=GRID)


def _assert_encodes_like(model, dense: dict[str, torch.Tensor]) -> None:
    hidden, vision = _run(model)
    expected_hidden, expected_vision = _run(_plain_reference(dense))
    torch.testing.assert_close(hidden, expected_hidden, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(vision, expected_vision, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("evidence", ["none", "marker", "header"])
@pytest.mark.parametrize("reason", ["fp8_compute", "fp8_storage"])
def test_a_device_that_holds_float8_keeps_every_scaled_linear_with_its_scale_and_encodes_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reason: str, evidence: str
) -> None:
    checkpoint = _write(tmp_path, evidence=evidence)
    _decide(monkeypatch, device="cuda:0", matmul=reason == "fp8_compute", storage=reason == "fp8_storage")
    attached: list[tuple[int, int]] = []
    warn = qwen_image.warn_on_unattached_scales
    monkeypatch.setattr(
        qwen_image,
        "warn_on_unattached_scales",
        lambda logger, what, count, layers: (attached.append((count, len(layers))), warn(logger, what, count, layers)),
    )

    model = _load(checkpoint)

    # Language model and vision tower (qkv, proj, mlp, merger) alike.
    assert any(path.startswith("model.visual.blocks.0.attn") for path in checkpoint.linear_modules)
    assert any(path.startswith("model.visual.merger") for path in checkpoint.linear_modules)
    for path in checkpoint.linear_modules:
        module = model.get_submodule(path)
        assert module.weight.dtype is FP8, path
        assert torch.equal(module.weight_scale.reshape(-1), checkpoint.scales[path].reshape(-1)), path
        assert "weight_scale" not in module.state_dict(), path
    # Everything else is the compute dtype: a lookup table or a convolution kept float8 would feed it to the next layer.
    for key, param in model.named_parameters():
        assert (param.dtype is FP8) == (key.removesuffix(".weight") in checkpoint.linear_modules), key
    embed = model.model.language_model.embed_tokens.weight
    assert torch.equal(embed, checkpoint.dense["model.language_model.embed_tokens.weight"])
    assert model.dtype is torch.float32 and model.model.visual.dtype is torch.float32
    assert attached == [(len(checkpoint.linear_modules),) * 2]

    apply_custom_layers_to_model(model)
    _assert_encodes_like(model, checkpoint.dense)


def test_kept_layers_are_dequantized_per_forward_even_where_the_fp8_matmul_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Quantizing the activations as well (down-projection and vision outliers) would change the conditioning for a
    speedup one encoder pass per prompt does not need."""
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, device="cuda:0", matmul=True)
    model = _load(checkpoint)
    apply_custom_layers_to_model(model)

    def refuse(*_args, **_kwargs):
        raise AssertionError("an encoder layer ran on the fp8 matmul")

    monkeypatch.setattr(custom_linear, "is_fp8_matmul_enabled", lambda: True)
    monkeypatch.setattr(custom_linear, "device_supports_fp8_matmul", lambda _device: True)
    monkeypatch.setattr(custom_linear, "scaled_mm_linear", refuse)

    assert all(model.get_submodule(path)._fp8_full_precision_matmul is True for path in checkpoint.linear_modules)
    _assert_encodes_like(model, checkpoint.dense)


def test_a_device_without_float8_folds_every_scaled_weight_to_what_the_file_decodes_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, device="cpu", matmul=True, storage=True)

    model = _load(checkpoint)

    assert all(param.dtype is torch.float32 for param in model.parameters())
    assert not any(hasattr(module, "weight_scale") for module in model.modules())
    for key, value in model.state_dict().items():
        # Bitwise, so a per-row scale applied along the wrong axis cannot hide behind a tolerance.
        assert torch.equal(value, checkpoint.dense[key]), key
    _assert_encodes_like(model, checkpoint.dense)


def test_an_unscaled_fp8_file_stays_fp8_where_float8_is_held_and_decodes_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path, scaled=False)

    _decide(monkeypatch, device="cuda:0", storage=True)
    kept = _load(checkpoint)
    for path in checkpoint.linear_modules:
        module = kept.get_submodule(path)
        assert module.weight.dtype is FP8 and not hasattr(module, "weight_scale"), path
        assert module._fp8_full_precision_matmul is True, path
    apply_custom_layers_to_model(kept)
    _assert_encodes_like(kept, checkpoint.dense)

    _decide(monkeypatch, device="cpu", storage=True)
    folded = _load(checkpoint)
    assert all(param.dtype is torch.float32 for param in folded.parameters())
    _assert_encodes_like(folded, checkpoint.dense)


@pytest.mark.parametrize("evidence", ["marker", "header"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_a_mxfp8_layer_named_by_its_marker_or_the_header_is_decoded_and_the_rest_keeps_its_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: str, device: str
) -> None:
    checkpoint = _write(tmp_path, evidence=evidence, mx_module=MX_MODULE)
    _decide(monkeypatch, device=device, storage=True)

    model = _load(checkpoint)

    # Block-wise scales never reach the matmul, so this layer is folded even where the others stay float8.
    assert model.get_submodule(MX_MODULE).weight.dtype is torch.float32
    assert torch.equal(model.get_submodule(MX_MODULE).weight, checkpoint.dense[f"{MX_MODULE}.weight"])
    untouched = model.get_submodule("model.language_model.layers.1.self_attn.q_proj")
    assert untouched.weight.dtype is (FP8 if device == "cuda:0" else torch.float32)
    if device == "cuda:0":
        apply_custom_layers_to_model(model)
    _assert_encodes_like(model, checkpoint.dense)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_an_mxfp8_layer_nothing_names_is_refused_before_the_cache_is_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    checkpoint = _write(tmp_path, evidence="none", mx_module=MX_MODULE)  # the grid is there; no marker, no header
    _decide(monkeypatch, device=device, storage=True)
    loader = _loader()

    with pytest.raises(NotImplementedError, match="MXFP8"):
        _load(checkpoint, loader)

    loader._ram_cache.make_room.assert_not_called()


def test_mxfp8_decode_is_refused_before_expanding_scales_when_host_room_is_exactly_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path, evidence="marker", mx_module=MX_MODULE)
    _decide(monkeypatch, device="cpu", storage=True)
    loader = _loader()
    decoded_bytes = (
        128 * 4 * (torch.float32.itemsize + torch.uint8.itemsize + 2 * torch.int32.itemsize + torch.float32.itemsize)
    )

    def make_room(_reserved: int, *, host_bytes_needed: int | None = None):
        return SimpleNamespace(ram_headroom_satisfied=host_bytes_needed != decoded_bytes)

    loader._ram_cache.make_room = MagicMock(side_effect=make_room)
    extracted = False
    original_extract = qwen_image.extract_fp8_scaled_layers

    def observe_extract(*args, **kwargs):
        nonlocal extracted
        extracted = True
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(qwen_image, "extract_fp8_scaled_layers", observe_extract)

    with pytest.raises(MemoryError, match="Insufficient host memory"):
        _load(checkpoint, loader)

    assert extracted is False
    assert loader._ram_cache.make_room.call_args.kwargs == {"host_bytes_needed": decoded_bytes}


def test_the_reservation_counts_float8_weights_one_byte_and_everything_else_at_the_compute_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path)
    reserved = {}
    for device in ("cuda:0", "cpu"):
        _decide(monkeypatch, device=device, storage=True)
        loader = _loader()
        _load(checkpoint, loader)
        loader._ram_cache.make_room.assert_called_once()
        reserved[device] = loader._ram_cache.make_room.call_args.args[0]

    # The recovered scales are resident until the split consumes them; the 1.0 `scale_input` placeholders are dropped.
    widened = 4 * checkpoint.other_elements + checkpoint.scale_bytes
    assert reserved["cuda:0"] == checkpoint.linear_elements + widened
    assert reserved["cpu"] == 4 * checkpoint.linear_elements + widened


@pytest.mark.parametrize("evidence", ["none", "marker", "header"])
def test_the_lm_head_in_the_file_is_neither_loaded_nor_reserved_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: str
) -> None:
    """The head's weight, its scale and whatever names it quantized (marker or header entry) all stay behind."""
    checkpoint = _write(tmp_path, evidence=evidence)
    with safe_open(checkpoint.path, framework="pt") as saved:
        assert {"lm_head.weight", "lm_head.scale_weight"} <= set(saved.keys())
    _decide(monkeypatch, device="cpu")
    loader = _loader()
    unexpected: list[str] = []
    monkeypatch.setattr(qwen_image, "log_unexpected_keys", lambda _source, keys: unexpected.extend(keys))

    model = _load(checkpoint, loader)

    assert list(model.lm_head.parameters()) == [] and list(model.lm_head.buffers()) == []
    assert not any("lm_head" in key for key in unexpected)
    widened = 4 * (checkpoint.linear_elements + checkpoint.other_elements) + checkpoint.scale_bytes
    loader._ram_cache.make_room.assert_called_once()
    call = loader._ram_cache.make_room.call_args
    assert call.args == (widened,)
    # Exact incremental peak for the fixed toy checkpoint; its file LM head is excluded from admission.
    assert call.kwargs == {"host_bytes_needed": 1_216_512}
    with pytest.raises(RuntimeError, match="without its LM head"):
        model(input_ids=IDS)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_a_scale_spelling_the_loader_does_not_read_is_refused_rather_than_cast_unscaled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    checkpoint = _write(tmp_path, extra={"model.layers.0.self_attn.q_proj.weight_scale_inv": torch.tensor(2.0)})
    _decide(monkeypatch, device=device, storage=True)
    loader = _loader()

    with pytest.raises(NotImplementedError, match="weight_scale_inv"):
        _load(checkpoint, loader)

    loader._ram_cache.make_room.assert_not_called()


def test_a_lora_on_a_kept_float8_layer_is_added_beside_the_scaled_weight_and_removed_afterwards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, device="cuda:0", storage=True)
    model = _load(checkpoint)
    apply_custom_layers_to_model(model)
    rank = 4
    torch.manual_seed(5)
    # As large as the decoded weights (codes up to 12), so the patch visibly moves the output.
    up, down = torch.randn(128, rank) * 2, torch.randn(rank, 128) * 2
    patch = ModelPatchRaw({Q_PROJ: LoRALayer(up=up, mid=None, down=down, alpha=float(rank), bias=None)})
    unpatched = _run(model)[0]

    with LayerPatcher.apply_smart_model_patches(model=model, patches=[(patch, 1.0)], prefix="", dtype=torch.float32):
        patched = _run(model)[0]
        # The base weight is untouched: the patch is a sidecar.
        assert model.get_submodule(Q_PROJ).weight.dtype is FP8

    # Independent of the sidecar code: the plain model with the delta written into the decoded q_proj weight.
    merged = dict(checkpoint.dense)
    merged[f"{Q_PROJ}.weight"] = merged[f"{Q_PROJ}.weight"] + up @ down
    torch.testing.assert_close(patched, _run(_plain_reference(merged))[0], rtol=1e-4, atol=1e-4)
    assert (patched - unpatched).norm() > 0.05 * unpatched.norm()
    torch.testing.assert_close(_run(model)[0], unpatched)


def test_the_cache_key_and_the_stored_format_come_from_one_capability_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The key is computed before the load lock and the build runs under it; a probe that turns positive in between
    # must not store float8 under a key that says folded, where an incapable device would adopt it.
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, device="cuda:0")
    loader = _loader()
    config = QwenVLEncoder_Checkpoint_Config.model_construct(path=str(checkpoint.path))
    assert loader._cache_key_variant(config, SubModelType.TextEncoder) == "full-float32"

    _decide(monkeypatch, device="cuda:0", storage=True)
    model = _load(checkpoint, loader, config)

    assert all(param.dtype is torch.float32 for param in model.parameters())


def test_cpu_encoder_uses_cpu_dtype_for_key_and_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = _write(tmp_path)
    _decide(monkeypatch, device="cpu", storage=True)
    monkeypatch.setattr(
        qwen_image.TorchDevice,
        "choose_bfloat16_safe_dtype",
        lambda device: torch.float32 if device.type == "cpu" else torch.bfloat16,
    )
    loader = _loader()
    config = QwenVLEncoder_Checkpoint_Config.model_construct(path=str(checkpoint.path))
    assert loader._cache_key_variant(config, SubModelType.TextEncoder) == "full-float32"
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float16)

    model = _load(checkpoint, loader, config)

    assert all(param.dtype is torch.float32 for param in model.parameters())


def _device_loader(store, budget, device: str, tmp_path: Path) -> QwenVLEncoderCheckpointLoader:
    """A loader on its own device cache that shares `store` with the others."""
    from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache

    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=InvokeAILogger.get_logger("test"),
        shared_cpu_weights=store,
        ram_budget=budget,
    )
    loader = object.__new__(QwenVLEncoderCheckpointLoader)
    loader._app_config = MagicMock(log_memory_usage=False, models_path=tmp_path)
    loader._logger = InvokeAILogger.get_logger("test")
    loader._ram_cache = cache
    loader._torch_device = torch.device(device)
    return loader


@pytest.mark.parametrize("capable_first", [True, False], ids=["capable_loads_first", "incapable_loads_first"])
def test_device_caches_share_weights_only_with_a_device_that_builds_the_same_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capable_first: bool
) -> None:
    # Device caches share one canonical CPU copy per cache key and adopt it without rerunning the loader. Whether the
    # Linear weights stay float8 depends on the device, so the second device must neither adopt float8 it cannot hold
    # nor the folded copy that throws away a capable device's saving. Two GPUs, only the first passing the storage probe.
    from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
    from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore

    checkpoint = _write(tmp_path)
    monkeypatch.setattr(qwen_image, "load_bundled_qwen2_5_vl_config_dict", _toy_config_dict)
    monkeypatch.setattr(qwen_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: False)
    monkeypatch.setattr(
        load_default, "_device_supports_fp8_storage", lambda d, _logger=None: d == torch.device("cuda:0")
    )
    config = QwenVLEncoder_Checkpoint_Config.model_construct(
        key="qwen-vl",
        path=checkpoint.path.name,
        name="qwen_2.5_vl_7b_fp8_scaled",
        base=BaseModelType.Any,
        type=ModelType.QwenVLEncoder,
        cpu_only=None,
    )
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    # Both caches stay alive: a device's cache entry holds the shared weights only while its cache lives.
    loaders = {"capable": _device_loader(store, budget, "cuda:0", tmp_path)}
    loaders["incapable"] = _device_loader(store, budget, "cuda:1", tmp_path)

    records = {}
    for name in ["capable", "incapable"] if capable_first else ["incapable", "capable"]:
        records[name], _claim = loaders[name]._load_and_cache(config, SubModelType.TextEncoder)

    capable, incapable = (records[name].cached_model.model for name in ("capable", "incapable"))
    # The vision tower's layers too: it is part of the text encoder's module and shares its cache entry.
    assert any(path.startswith("model.visual.") for path in checkpoint.linear_modules)
    for path in checkpoint.linear_modules:
        kept = capable.get_submodule(path)
        assert kept.weight.dtype is FP8, path
        assert torch.equal(kept.weight_scale.reshape(-1), checkpoint.scales[path].reshape(-1)), path
        folded = incapable.get_submodule(path).weight
        assert folded.dtype is torch.float32 and torch.equal(folded, checkpoint.dense[f"{path}.weight"]), path
    assert records["capable"].key != records["incapable"].key
