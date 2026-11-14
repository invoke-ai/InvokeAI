"""Review regressions through native loaders and real shared CPU allocations; no accelerator execution."""

from contextlib import ExitStack

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import FP8_COMPUTE_DTYPE_ATTR
from tests.backend.model_manager.load.test_fp8_shared_weight_variants import FP8, _checkpoint_case, _loader
from tests.backend.model_manager.load.test_layerwise_storage_policy import _public_load


def _device_loaders(cls, tmp_path, devices):
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    return [_loader(cls, store, budget, device, tmp_path) for device in devices]


@pytest.mark.parametrize("family", ["mistral", "qwen", "krea", "ideogram"])
@pytest.mark.parametrize("reverse", [False, True])
def test_checkpoint_storage_without_fp8_compute_partitions_and_adopts(family, reverse, monkeypatch, tmp_path):
    cls, config, subtype, reads = _checkpoint_case(family, monkeypatch)
    (tmp_path / "tiny.safetensors").touch()
    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder, qwen_image

    # M1 imported probe aliases in these modules. The same fixtures cover both native implementations.
    for module in (load_default, mistral_encoder, qwen_image):
        if hasattr(module, "should_keep_fp8_weights"):
            monkeypatch.setattr(module, "should_keep_fp8_weights", lambda _device: False)
    for module in (load_default, mistral_encoder):
        if hasattr(module, "_device_supports_fp8_storage"):
            monkeypatch.setattr(module, "_device_supports_fp8_storage", lambda device, _logger: device.index == 0)
    capable, incapable, peer, folded_peer = _device_loaders(cls, tmp_path, ["cuda:0", "cuda:1", "cuda:0", "cuda:1"])
    settings = MainModelDefaultSettings(fp8_storage=True)
    with ExitStack() as stack:
        for loader in (capable, incapable, peer, folded_peer):
            stack.callback(loader._ram_cache.shutdown)
        order = [incapable, capable] if reverse else [capable, incapable]
        models = {loader: _public_load(loader, config(default_settings=settings), subtype) for loader in order}
        assert models[capable].proj.weight.dtype == FP8
        assert models[incapable].proj.weight.dtype == torch.float32
        torch.testing.assert_close(models[incapable].proj.weight, torch.full((4, 4), 0.5))
        assert models[capable].proj.weight.data_ptr() != models[incapable].proj.weight.data_ptr()
        assert len(reads) == 2
        adopted = _public_load(peer, config(default_settings=settings), subtype)
        assert adopted.proj.weight.data_ptr() == models[capable].proj.weight.data_ptr()
        folded = _public_load(folded_peer, config(default_settings=settings), subtype)
        assert folded.proj.weight.data_ptr() == models[incapable].proj.weight.data_ptr()
        assert len(reads) == 2


def _qwen_vl_checkpoint(monkeypatch, tmp_path):
    from transformers import Qwen3VLModel

    from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Checkpoint_Config
    from invokeai.backend.model_manager.load.model_loaders import krea2
    from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType
    from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config

    geometry = tiny_qwen3vl_config()
    reference = Qwen3VLModel(geometry)
    weights = {name: value.detach().clone().contiguous() for name, value in reference.state_dict().items()}
    weight = "language_model.layers.0.self_attn.q_proj.weight"
    weights[weight] = torch.ones_like(weights[weight]).to(FP8)
    weights[weight.removesuffix(".weight") + ".weight_scale"] = torch.tensor(0.5)
    path = tmp_path / "qwen-vl.safetensors"
    save_file(weights, str(path))
    monkeypatch.setattr(krea2.Qwen3VLEncoderCheckpointLoader, "_load_te_config", lambda *_args: geometry)
    reads = []
    import safetensors.torch

    original = safetensors.torch.load_file

    def read(*args, **kwargs):
        reads.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "load_file", read)

    def config(**extra):
        return Qwen3VLEncoder_Checkpoint_Config.model_construct(
            key="qwen-vl", name="qwen-vl", path=str(path), variant=Qwen3VLVariantType.Qwen3VL_4B, **extra
        )

    return krea2.Qwen3VLEncoderCheckpointLoader, config, weight, reads


@pytest.mark.parametrize("consumer", ["compute", "storage"])
@pytest.mark.parametrize("reverse", [False, True])
def test_issue_qwen_vl_native_body_preserves_scales_and_adopts(consumer, reverse, monkeypatch, tmp_path):
    from invokeai.backend.model_manager.load.model_loaders import krea2

    cls, config, weight, reads = _qwen_vl_checkpoint(monkeypatch, tmp_path)
    for module in (load_default, krea2):
        if hasattr(module, "should_keep_fp8_weights"):
            monkeypatch.setattr(module, "should_keep_fp8_weights", lambda d: consumer == "compute" and d.index == 0)
        if hasattr(module, "_device_supports_fp8_storage"):
            monkeypatch.setattr(
                module, "_device_supports_fp8_storage", lambda d, _logger: consumer == "storage" and d.index == 0
            )
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))
    capable, incapable, peer, folded_peer = _device_loaders(cls, tmp_path, ["cuda:0", "cuda:1", "cuda:0", "cuda:1"])
    with ExitStack() as stack:
        for loader in (capable, incapable, peer, folded_peer):
            stack.callback(loader._ram_cache.shutdown)
        order = [incapable, capable] if reverse else [capable, incapable]
        models = {loader: _public_load(loader, config(), SubModelType.TextEncoder) for loader in order}
        packed, folded = (models[loader].get_parameter(weight) for loader in (capable, incapable))
        assert packed.dtype == FP8 and folded.dtype == torch.float32
        torch.testing.assert_close(folded, torch.full_like(folded, 0.5))
        torch.testing.assert_close(
            models[capable].language_model.layers[0].self_attn.q_proj.weight_scale, torch.tensor(0.5)
        )
        assert packed.data_ptr() != folded.data_ptr()
        assert len(reads) == 2
        adopted = _public_load(peer, config(), SubModelType.TextEncoder)
        assert adopted.get_parameter(weight).data_ptr() == packed.data_ptr()
        adopted_folded = _public_load(folded_peer, config(), SubModelType.TextEncoder)
        assert adopted_folded.get_parameter(weight).data_ptr() == folded.data_ptr()
        assert len(reads) == 2


def test_cpu_only_qwen_vl_folds_and_shares_with_cpu(monkeypatch, tmp_path):
    from invokeai.backend.model_manager.load.model_loaders import krea2

    cls, config, weight, reads = _qwen_vl_checkpoint(monkeypatch, tmp_path)
    for module in (load_default, krea2):
        if hasattr(module, "should_keep_fp8_weights"):
            monkeypatch.setattr(module, "should_keep_fp8_weights", lambda d: d.type == "cuda")
        if hasattr(module, "_device_supports_fp8_storage"):
            monkeypatch.setattr(module, "_device_supports_fp8_storage", lambda d, _logger: d.type == "cuda")
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))
    cpu, cpu_only, gpu = _device_loaders(cls, tmp_path, ["cpu", "cuda:0", "cuda:0"])
    with ExitStack() as stack:
        for loader in (cpu, cpu_only, gpu):
            stack.callback(loader._ram_cache.shutdown)
        a = _public_load(cpu, config(), SubModelType.TextEncoder).get_parameter(weight)
        b = _public_load(cpu_only, config(cpu_only=True), SubModelType.TextEncoder).get_parameter(weight)
        c = _public_load(gpu, config(), SubModelType.TextEncoder).get_parameter(weight)
        assert a.dtype == b.dtype == torch.float32
        assert c.dtype == FP8
        assert a.data_ptr() == b.data_ptr() != c.data_ptr()
        torch.testing.assert_close(a, torch.full_like(a, 0.5))
        assert len(reads) == 2


def _native_dtype_case(family, monkeypatch, tmp_path):
    from invokeai.backend.model_manager.configs import main
    from invokeai.backend.model_manager.load.model_loaders import flux3, ltx2, wan

    path = tmp_path / family
    if family.startswith("flux3"):
        from tests.backend.flux3.flux3_fixture import tiny_flux3_model, to_native

        reference = tiny_flux3_model(False)
        if family == "flux3-folder":
            reference.save_pretrained(path / "transformer")
            cls, config_cls = flux3.Flux3DiffusersModel, main.Main_Diffusers_Flux3_Config
        else:
            save_file(to_native(reference), str(path))
            cls, config_cls = flux3.Flux3CheckpointModel, main.Main_Checkpoint_Flux3_Config
    elif family == "wan":
        from tests.backend.model_manager.load.test_wan_checkpoint_loader import _tiny_model

        save_file(_tiny_model().state_dict(), str(path))
        cls, config_cls = wan.WanCheckpointModel, main.Main_Checkpoint_Wan_Config
    else:
        from invokeai.backend.ltx2 import component_configs as cc
        from tests.backend.model_manager.load.ltx2_transformer_fixture import (
            TINY_TRANSFORMER,
            tiny_transformer_checkpoint,
        )

        monkeypatch.setattr(cc, "LTX2_5_TRANSFORMER_CONFIG", TINY_TRANSFORMER)
        save_file(tiny_transformer_checkpoint(), str(path))
        cls, config_cls = ltx2.LTX2CheckpointModel, main.Main_Checkpoint_LTX2_Config

    def config():
        return config_cls.model_construct(
            key="dtype", name="dtype", path=str(path), default_settings=MainModelDefaultSettings(fp8_storage=True)
        )

    return cls, config


@pytest.mark.parametrize("family", ["flux3", "flux3-folder", "wan", "ltx2"])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_constructor_key_matches_real_bfloat_rejection_fallback(family, reverse, monkeypatch, tmp_path):
    cls, config = _native_dtype_case(family, monkeypatch, tmp_path)
    empty = torch.empty
    probes = []

    def reject_bfloat(*shape, **kwargs):
        device = kwargs.get("device")
        if shape == (0,) and kwargs.get("dtype") == torch.bfloat16 and device is not None:
            device = torch.device(device)
            if device.type in ("cuda", "xpu"):
                probes.append(str(device))
                raise TypeError("fixture backend rejects bfloat16")
        return empty(*shape, **kwargs)

    # Exercise the real choose_bfloat16_safe_dtype, including its device-specific TypeError fallback.
    monkeypatch.setattr(torch, "empty", reject_bfloat)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_args: True)
    half, full, peer = _device_loaders(cls, tmp_path, ["cuda:0", "xpu:0", "cuda:1"])
    for loader in (half, full, peer):
        loader._torch_dtype = torch.float16
    with ExitStack() as stack:
        for loader in (half, full, peer):
            stack.callback(loader._ram_cache.shutdown)
        order = [full, half] if reverse else [half, full]
        models = {loader: _public_load(loader, config(), SubModelType.Transformer) for loader in order}
        for loader, dtype in ((half, torch.float16), (full, torch.float32)):
            assert getattr(models[loader], FP8_COMPUTE_DTYPE_ATTR) == dtype
            assert str(dtype).removeprefix("torch.") in loader._get_cache_key(config(), SubModelType.Transformer)
        a, b = (next(model.parameters()) for model in (models[half], models[full]))
        assert a.data_ptr() != b.data_ptr()
        adopted = _public_load(peer, config(), SubModelType.Transformer)
        assert next(adopted.parameters()).data_ptr() == a.data_ptr()
        assert {"cuda:0", "xpu:0"} <= set(probes)
