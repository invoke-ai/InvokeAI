"""Guards for Anima's FP8-storage wiring.

Two things have to hold for the `fp8_storage` toggle to do the right thing on Anima, and neither
is observed by any other test:

1. The loader has to actually call `_apply_fp8_layerwise_casting`. Without it the toggle renders
   in the UI and silently does nothing.
2. `AnimaTransformer._skip_layerwise_casting_patterns` has to match the modules it is meant to
   match. The patterns are plain substrings matched against dotted module paths, so a rename in
   the transformer disables the skip without breaking anything loudly.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import accelerate
import pytest
import safetensors.torch
import torch

from invokeai.backend.anima.anima_transformer import AnimaTransformer
from invokeai.backend.anima.checkpoint_layout import ANIMA_TRANSFORMER_CONFIG
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config, MainModelDefaultSettings
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.load_default import (
    _FP8_DEFAULT_SKIP_PATTERNS,
    _FP8_SUPPORTED_PYTORCH_LAYERS,
)
from invokeai.backend.model_manager.load.model_loaders.anima import AnimaCheckpointModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from tests.fixtures.loader_seams import assert_fp8_storage_declined, logged_info
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot
from tests.fixtures.safetensors_files import record_reads, served_checkpoint

# Every cast-eligible module in the real Anima graph that the declared patterns protect. Pinned
# exhaustively rather than as a substring check, so both a rename (entries disappear) and an
# over-broad pattern (extra entries appear) fail here.
EXPECTED_SKIPPED_MODULES = {
    "t_embedder.1.linear_1",
    "t_embedder.1.linear_2",
    "x_embedder.proj.1",
    "final_layer.linear",
    "final_layer.adaln_modulation.1",
    "final_layer.adaln_modulation.2",
}


def _cast_eligible_modules(model: torch.nn.Module) -> list[str]:
    """The dotted paths `_apply_fp8_to_nn_module` would consider for casting, before skipping."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, _FP8_SUPPORTED_PYTORCH_LAYERS) and list(module.parameters(recurse=False))
    ]


def _build_meta_transformer() -> AnimaTransformer:
    with accelerate.init_empty_weights():
        return AnimaTransformer(**ANIMA_TRANSFORMER_CONFIG)


def test_declared_skip_patterns_pin_to_real_module_paths() -> None:
    """The declared patterns must resolve against the real module graph, not just be strings.

    `t_embedder` is the load-bearing one: casting it to FP8 renders a heavily dithered image,
    because it feeds `adaln_lora` into every block. Renaming it in `AnimaTransformer` while the
    pattern list still says `t_embedder` would silently reintroduce that, so match against an
    actually-instantiated model.
    """
    model = _build_meta_transformer()
    patterns = AnimaTransformer._skip_layerwise_casting_patterns

    skipped = {name for name in _cast_eligible_modules(model) if any(re.search(p, name) for p in patterns)}

    assert skipped == EXPECTED_SKIPPED_MODULES

    # Each declared pattern earns its place — none is dead.
    for pattern in patterns:
        assert any(re.search(pattern, name) for name in skipped), f"pattern {pattern!r} matches nothing"


def test_generic_skip_patterns_do_not_cover_anima() -> None:
    """The declared list is not redundant with `_FP8_DEFAULT_SKIP_PATTERNS`.

    Those defaults are written for diffusers' module naming (`norm`, `pos_embed`, `patch_embed`,
    `proj_in/out`); this architecture names the equivalent modules differently, so they protect
    nothing here. If a future default did start covering Anima, this test failing is the signal to
    re-check whether the declared list is still needed.
    """
    model = _build_meta_transformer()

    covered_by_defaults = [
        name for name in _cast_eligible_modules(model) if any(re.search(p, name) for p in _FP8_DEFAULT_SKIP_PATTERNS)
    ]

    assert covered_by_defaults == []


class _TinyAnimaTransformer(torch.nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 2))


def test_single_file_loader_applies_fp8_layerwise_casting(monkeypatch, tmp_path) -> None:
    """Regression guard for the `fp8_storage` toggle being wired up at all.

    Deleting the `_apply_fp8_layerwise_casting` call from `_load_from_singlefile` leaves the whole
    `tests/backend/model_manager` and `tests/backend/anima` suites green — nothing else observes
    it — so the dead toggle can come straight back with CI passing.
    """

    import invokeai.backend.anima.anima_transformer as anima_transformer_module
    import invokeai.backend.model_manager.load.model_loaders.anima as loader_module

    checkpoint_path = tmp_path / "anima.safetensors"
    # The loader seam replaces the payload, but ownership selection still reads the real header.
    safetensors.torch.save_file({"_fixture": torch.ones(1)}, checkpoint_path)
    # `fp8_storage` lives under `default_settings` -- passing it as a top-level kwarg to
    # `model_construct` would be silently discarded (the config has no such field and no extra="allow"),
    # leaving the toggle off in a test whose whole point is that the toggle is wired up.
    config = Main_Checkpoint_Anima_Config.model_construct(
        path=str(checkpoint_path), default_settings=MainModelDefaultSettings(fp8_storage=True)
    )

    monkeypatch.setattr(anima_transformer_module, "AnimaTransformer", _TinyAnimaTransformer)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.load_safetensors_checkpoint",
        served_checkpoint({"weight": torch.ones(2, 2)}),
    )
    # The stand-in state dict has no DiT blocks to read a depth from, and the tiny model takes no config.
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.anima_transformer_config", lambda _sd: {}
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_anima_inference_dtype",
        lambda _device: torch.float32,
    )

    reservations: list[dict[str, object]] = []
    reserve_for_load = loader_module.reserve_for_load

    def record_reservation(*args: object, **kwargs: object) -> None:
        reservations.append(kwargs)
        reserve_for_load(*args, **kwargs)

    monkeypatch.setattr(loader_module, "reserve_for_load", record_reservation)

    cast_calls: list[tuple[object, object]] = []

    def _record_cast(model, cfg, submodel):
        cast_calls.append((cfg, submodel))
        return model

    loader = object.__new__(AnimaCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._apply_fp8_layerwise_casting = _record_cast
    # The real loader gets both from `ModelLoader.__init__`; `_keep_fp8_weights` reads them to ask
    # whether this device can hold fp8 at all.
    loader._torch_device = torch.device("cpu")
    loader._logger = MagicMock()

    model = loader._load_from_singlefile(config)

    assert isinstance(model, _TinyAnimaTransformer)
    assert torch.equal(model.weight, torch.ones(2, 2))
    assert cast_calls == [(config, SubModelType.Transformer)]
    assert reservations[0]["source_tensors_mapped"] is True


def test_mxfp8_decode_headroom_is_checked_before_extraction(monkeypatch, tmp_path) -> None:
    import invokeai.backend.anima.anima_transformer as anima_transformer_module
    import invokeai.backend.model_manager.load.model_loaders.anima as loader_module

    checkpoint_path = tmp_path / "anima_mxfp8.safetensors"
    safetensors.torch.save_file({"_fixture": torch.ones(1)}, checkpoint_path)
    config = Main_Checkpoint_Anima_Config.model_construct(path=str(checkpoint_path))
    monkeypatch.setattr(anima_transformer_module, "AnimaTransformer", _TinyAnimaTransformer)
    monkeypatch.setattr(loader_module, "anima_transformer_config", lambda _sd: {})
    monkeypatch.setattr(loader_module, "load_safetensors_checkpoint", served_checkpoint({"weight": torch.ones(2, 2)}))
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loader_module.TorchDevice, "choose_anima_inference_dtype", lambda _device: torch.float32)
    monkeypatch.setattr(
        "invokeai.backend.quantization.fp8_scaled.mxfp8_scale_decode_host_bytes", lambda _sd, _hints: 1234
    )

    loader = object.__new__(AnimaCheckpointModel)
    make_room = MagicMock(return_value=SimpleNamespace(ram_headroom_satisfied=False))
    loader._ram_cache = SimpleNamespace(make_room=make_room)
    loader._torch_device = torch.device("cpu")
    loader._logger = MagicMock()
    extracted = False

    def observe_extract(*args, **kwargs):
        nonlocal extracted
        extracted = True
        return {}

    monkeypatch.setattr(loader_module, "extract_fp8_scaled_layers", observe_extract)
    with pytest.raises(MemoryError, match="Insufficient host memory"):
        loader._load_from_singlefile(config)

    assert extracted is False
    make_room.assert_called_once_with(0, host_bytes_needed=1234)


def test_single_file_loader_passes_mapped_source_to_int8_installer(monkeypatch, tmp_path) -> None:
    """The int8 installer must receive the same already-mapped source contract as reservation."""
    import invokeai.backend.anima.anima_transformer as anima_transformer_module
    import invokeai.backend.model_manager.load.model_loaders.anima as loader_module

    checkpoint_path = tmp_path / "anima.safetensors"
    safetensors.torch.save_file({"weight": torch.ones(2, 2)}, checkpoint_path)
    config = Main_Checkpoint_Anima_Config.model_construct(path=str(checkpoint_path))
    monkeypatch.setattr(anima_transformer_module, "AnimaTransformer", _TinyAnimaTransformer)
    monkeypatch.setattr(loader_module, "anima_transformer_config", lambda _sd: {})
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(loader_module.TorchDevice, "choose_anima_inference_dtype", lambda _device: torch.float32)
    monkeypatch.setattr(loader_module, "extract_int8_convrot_markers", lambda _sd: {"weight": {"format": "int8"}})
    monkeypatch.setattr(loader_module, "reject_unmarked_int8_weights", lambda *_args: None)

    install_calls: list[dict[str, object]] = []

    def _record_install(_model, _sd, quantized, _dtype, **kwargs):
        install_calls.append(kwargs)
        return quantized

    monkeypatch.setattr(loader_module, "install_int8_convrot_layers", _record_install)
    loader = object.__new__(AnimaCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model
    loader._torch_device = torch.device("cpu")
    loader._logger = MagicMock()

    model = loader._load_from_singlefile(config)

    assert isinstance(model, _TinyAnimaTransformer)
    assert len(install_calls) == 1
    assert install_calls[0]["source_tensors_mapped"] is True


class _TinyInt8AnimaTransformer(torch.nn.Module):
    """One layer an int8 build quantizes beside one it leaves dense, under the names its markers use."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)
        self.embed = torch.nn.Linear(8, 4)


def test_fp8_storage_leaves_an_int8_build_as_loaded(monkeypatch, tmp_path) -> None:
    """The int8 layer stays int8 and the dense one keeps the compute dtype: the build is the same tensor for tensor
    as the load without FP8 Storage."""
    import invokeai.backend.anima.anima_transformer as anima_transformer_module

    quantized = quantize_convrot(torch.randn(4, CONVROT_GROUP_SIZE))
    dense = {"embed.weight": torch.randn(4, 8), "embed.bias": torch.randn(4)}
    state_dict = {
        "proj.weight": quantized.codes,
        "proj.weight_scale": quantized.scale,
        "proj.comfy_quant": comfy_quant_marker(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}
        ),
        **dense,
    }
    checkpoint_path = tmp_path / "anima_int8.safetensors"
    safetensors.torch.save_file(state_dict, str(checkpoint_path))
    monkeypatch.setattr(anima_transformer_module, "AnimaTransformer", _TinyInt8AnimaTransformer)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.anima_transformer_config", lambda _sd: {}
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_anima_inference_dtype",
        lambda _device: torch.bfloat16,
    )
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_args: True)

    def load(fp8_storage: bool):
        config = Main_Checkpoint_Anima_Config.model_construct(
            path=str(checkpoint_path),
            name="anima_int8",
            default_settings=MainModelDefaultSettings(fp8_storage=fp8_storage),
        )
        loader = object.__new__(AnimaCheckpointModel)
        loader._ram_cache = SimpleNamespace(make_room=MagicMock())
        loader._torch_device = torch.device("cpu")
        loader._torch_dtype = torch.bfloat16
        loader._logger = MagicMock()
        return loader._load_from_singlefile(config), loader

    plain, _ = load(False)
    model, loader = load(True)

    assert isinstance(model.proj, Int8ConvrotLinear)
    assert model.proj.weight.dtype is torch.int8
    assert model.embed.weight.dtype is torch.bfloat16
    assert_fp8_storage_declined(model, plain, logged_info(loader._logger), "anima_int8")


def test_an_all_in_ones_vae_text_encoder_and_exporter_metadata_are_never_read(monkeypatch, tmp_path) -> None:
    """A ComfyUI bundle carries the VAE and text encoder under namespaces of their own, and exporters add sampling
    schedules and derived buffers. Read through ``pread`` they would cost their size in process memory only to be
    dropped."""
    import invokeai.backend.anima.anima_transformer as anima_transformer_module

    transformer = {"model.diffusion_model.weight": torch.ones(2, 2).to(torch.float8_e4m3fn)}
    elsewhere = {
        "first_stage_model.decoder.conv.weight": torch.ones(8, 8),
        "cond_stage_model.layers.0.weight": torch.ones(8, 8),
        "model.diffusion_model.pos_embedder.seq": torch.arange(4.0),
        "model.diffusion_model.blocks.0.rope.inv_freq": torch.ones(4),
        "model_sampling.sigmas": torch.ones(4),
    }
    checkpoint_path = tmp_path / "anima.safetensors"
    safetensors.torch.save_file({**transformer, **elsewhere}, str(checkpoint_path))
    monkeypatch.setattr(anima_transformer_module, "AnimaTransformer", _TinyAnimaTransformer)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.anima_transformer_config", lambda _sd: {}
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.anima.TorchDevice.choose_anima_inference_dtype",
        lambda _device: torch.float32,
    )
    loader = object.__new__(AnimaCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model
    loader._torch_device = torch.device("cpu")
    loader._logger = MagicMock()
    log = record_reads(monkeypatch)

    loader._load_from_singlefile(
        Main_Checkpoint_Anima_Config.model_construct(
            path=str(checkpoint_path), default_settings=MainModelDefaultSettings()
        )
    )

    assert set(log.tensors) == set(transformer)
