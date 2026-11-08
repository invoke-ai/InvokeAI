"""Native quantized loading and device policy share one captured decision; tensors stay on CPU."""

from contextlib import ExitStack

import gguf
import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux3_Config, Main_GGUF_Flux3_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import flux3
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.util.devices import TorchDevice
from tests.backend.flux3.flux3_fixture import (
    QUANTIZED_SCHEMES,
    quantize_flux3_native,
    tiny_flux3_model,
    to_native,
    write_flux3_gguf,
)
from tests.backend.model_manager.load.test_fp8_review_regressions import _device_loaders
from tests.backend.model_manager.load.test_layerwise_storage_policy import _public_load


@pytest.mark.parametrize("scheme", ["raw_fp8", *QUANTIZED_SCHEMES])
@pytest.mark.parametrize("consumer", ["compute", "storage"])
def test_native_policy_partitions_partial_devices_and_adopts_matching_cpu_weights(
    scheme, consumer, monkeypatch, tmp_path
):
    native = to_native(tiny_flux3_model(True))
    weight = "single_blocks.0.q_proj.weight"
    if scheme == "raw_fp8":
        native[weight] = native[weight].to(torch.float8_e4m3fn)
        metadata = {}
    else:
        build = quantize_flux3_native(native, scheme)
        native, metadata = build.state_dict, build.metadata
    path = tmp_path / "native.safetensors"
    safetensors.torch.save_file(native, str(path), metadata=metadata)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda d: consumer == "compute" and d.index == 0)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda d, _logger: d.index == 0)
    calls = []
    factory = flux3.flux3_transformer_from_native

    def build(sd, dtype, reserve, **kwargs):
        calls.append((dtype, kwargs["keep_fp8"], kwargs["metadata"]))
        return factory(sd, dtype, reserve, **kwargs)

    monkeypatch.setattr(flux3, "flux3_transformer_from_native", build)
    capable, incapable, peer = _device_loaders(flux3.Flux3CheckpointModel, tmp_path, ["cuda:0", "cuda:1", "cuda:0"])

    def config():
        return Main_Checkpoint_Flux3_Config.model_construct(
            key="native",
            name="native",
            path=str(path),
            default_settings=MainModelDefaultSettings(fp8_storage=consumer == "storage"),
        )

    with ExitStack() as stack:
        for loader in (capable, incapable, peer):
            stack.callback(loader._ram_cache.shutdown)
        folded = _public_load(incapable, config(), SubModelType.Transformer)
        kept = _public_load(capable, config(), SubModelType.Transformer)
        adopted = _public_load(peer, config(), SubModelType.Transformer)
        assert calls == [(torch.float32, False, metadata), (torch.float32, True, metadata)]
        name = "single_transformer_blocks.0.attn.to_q.weight"
        a, b, c = (model.get_submodule(name.removesuffix(".weight")).weight for model in (folded, kept, adopted))
        assert a.data_ptr() != b.data_ptr() == c.data_ptr()
        if scheme in ("fp8r", "raw_fp8"):
            assert a.dtype == torch.float32
            assert b.dtype == torch.float8_e4m3fn
            # This compares the decoded values, with the real checkpoint scale when present.
            expected = b.float()
            if scheme == "fp8r":
                expected = expected * kept.get_submodule(name.removesuffix(".weight")).weight_scale[:, None]
            torch.testing.assert_close(a, expected)
        else:
            # int8/nvfp4 remain packed even on the device which cannot retain checkpoint fp8.
            assert a.dtype == b.dtype == (torch.int8 if scheme == "int8_convrot" else torch.uint8)
            assert folded.x_embedders["image"].weight.dtype == kept.x_embedders["image"].weight.dtype == torch.float32


def test_native_factory_uses_the_key_snapshot_after_bfloat_rejection(monkeypatch, tmp_path):
    path = tmp_path / "native.safetensors"
    native = to_native(tiny_flux3_model(True))
    native["single_blocks.0.q_proj.weight"] = native["single_blocks.0.q_proj.weight"].to(torch.float8_e4m3fn)
    safetensors.torch.save_file(native, str(path))
    empty = torch.empty
    probes = []

    def reject_bfloat(*shape, **kwargs):
        if shape == (0,) and kwargs.get("dtype") == torch.bfloat16:
            probes.append(str(kwargs.get("device")))
            raise TypeError("fixture device rejects bfloat16")
        return empty(*shape, **kwargs)

    monkeypatch.setattr(torch, "empty", reject_bfloat)
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda d: d == torch.device("cuda:1"))
    (loader,) = _device_loaders(flux3.Flux3CheckpointModel, tmp_path, ["cuda:1"])
    config = Main_Checkpoint_Flux3_Config.model_construct(key="snapshot", name="native", path=str(path))
    try:
        key = loader._get_cache_key(config, SubModelType.Transformer)
        assert "fp8-float16" in key
        # A different session default must not change construction or redo the dtype/capability probes.
        monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _d: False)
        model = loader._load_model(config, SubModelType.Transformer)
        assert probes == ["cuda:1"]
        assert model.x_embedders["image"].weight.dtype == torch.float16
        assert model.single_transformer_blocks[0].attn.to_q.weight.dtype == torch.float8_e4m3fn
        assert loader._get_cache_key(config, SubModelType.Transformer) == key
    finally:
        loader._ram_cache.shutdown()


def test_gguf_compute_dtype_partitions_shared_weights_after_bfloat_rejection(monkeypatch, tmp_path):
    path = tmp_path / "native.gguf"
    write_flux3_gguf(path, to_native(tiny_flux3_model(True)), qtype=gguf.GGMLQuantizationType.Q8_0)
    empty = torch.empty
    probes = []

    def reject_bfloat(*shape, **kwargs):
        if shape == (0,) and kwargs.get("dtype") == torch.bfloat16:
            probes.append(str(kwargs.get("device")))
            raise TypeError("fixture device rejects bfloat16")
        return empty(*shape, **kwargs)

    monkeypatch.setattr(torch, "empty", reject_bfloat)
    half, full, peer = _device_loaders(flux3.Flux3GGUFCheckpointModel, tmp_path, ["cuda:0", "xpu:0", "cuda:1"])

    def config():
        return Main_GGUF_Flux3_Config.model_construct(key="gguf", name="native", path=str(path))

    with ExitStack() as stack:
        for loader in (half, full, peer):
            stack.callback(loader._ram_cache.shutdown)
        a = _public_load(half, config(), SubModelType.Transformer)
        b = _public_load(full, config(), SubModelType.Transformer)
        c = _public_load(peer, config(), SubModelType.Transformer)
        name = "single_transformer_blocks.0.attn.to_q.weight"
        wa, wb, wc = (model.get_parameter(name) for model in (a, b, c))
        assert wa.compute_dtype == wc.compute_dtype == torch.float16
        assert wb.compute_dtype == torch.float32
        assert wa.quantized_data.data_ptr() == wc.quantized_data.data_ptr() != wb.quantized_data.data_ptr()
        assert probes == ["cuda:0", "xpu:0", "cuda:1"]
