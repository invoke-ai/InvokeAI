"""Real CPU-resident GGUF weights with simulated execution-device policies."""

import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_GGUF_ErnieImage_Config, Main_GGUF_SD3_Config
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_loaders import ernie_image, sd3
from invokeai.backend.model_manager.load.model_loaders.ernie_image import ErnieImageGGUFModel
from invokeai.backend.model_manager.load.model_loaders.sd3 import SD3GGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.util.devices import TorchDevice
from tests.backend.model_manager.load.ernie_image_gguf_fixture import TINY_CONFIG, write_ernie_image_gguf
from tests.backend.model_manager.load.sd3_gguf_fixture import write_sd3_gguf
from tests.backend.model_manager.load.test_fp8_shared_weight_variants import _load, _loader


@pytest.mark.parametrize("family", ["ernie", "sd3"])
@pytest.mark.parametrize("reverse", [False, True])
def test_gguf_sharing_respects_frozen_execution_dtype(monkeypatch, tmp_path, family, reverse):
    policy = {"cuda:0": torch.float16, "cuda:1": torch.float32, "cpu": torch.float32}

    def choose(device):
        return policy[str(device)]

    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(choose))
    path = tmp_path / f"{family}.gguf"
    if family == "ernie":
        write_ernie_image_gguf(path)
        monkeypatch.setattr(ernie_image, "ERNIE_IMAGE_TRANSFORMER_CONFIG", TINY_CONFIG)
        cls, config_cls, module = ErnieImageGGUFModel, Main_GGUF_ErnieImage_Config, ernie_image
        packed_name = "layers.0.mlp.gate_proj.weight"
        dense_name = "x_embedder.proj.weight"
    else:
        write_sd3_gguf(path)
        cls, config_cls, module = SD3GGUFModel, Main_GGUF_SD3_Config, sd3
        packed_name = "transformer_blocks.0.attn.to_q.weight"
        dense_name = "pos_embed.proj.weight"
    config = config_cls.model_construct(key=family, name=family, path=str(path))
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    half = _loader(cls, store, budget, "cuda:0", tmp_path)
    full = _loader(cls, store, budget, "cuda:1", tmp_path)
    peer = _loader(cls, store, budget, "cuda:0", tmp_path)
    reads = []
    reader = module.gguf_sd_loader

    def read(*args, **kwargs):
        reads.append(kwargs["compute_dtype"])
        return reader(*args, **kwargs)

    monkeypatch.setattr(module, "gguf_sd_loader", read)
    try:
        order = [full, half] if reverse else [half, full]
        models = {
            loader: _load(
                loader, config, None if family == "ernie" and loader is half and reverse else SubModelType.Transformer
            )
            for loader in order
        }
        a = models[half].get_parameter(packed_name).data
        b = models[full].get_parameter(packed_name).data
        assert isinstance(a, GGMLTensor) and isinstance(b, GGMLTensor)
        assert a.compute_dtype == torch.float16 and b.compute_dtype == torch.float32
        assert models[half].get_parameter(dense_name).dtype == torch.float16
        assert models[full].get_parameter(dense_name).dtype == torch.float32
        assert a.quantized_data.data_ptr() != b.quantized_data.data_ptr()
        assert half._get_cache_key(config, SubModelType.Transformer) != full._get_cache_key(
            config, SubModelType.Transformer
        )
        adopted = _load(peer, config, None if family == "ernie" and not reverse else SubModelType.Transformer)
        c = adopted.get_parameter(packed_name).data
        assert c.compute_dtype == torch.float16
        assert c.quantized_data.data_ptr() == a.quantized_data.data_ptr()
        assert len(reads) == 2
    finally:
        for loader in (half, full, peer):
            loader._ram_cache.shutdown()


@pytest.mark.parametrize("family", ["ernie", "sd3"])
def test_gguf_cpu_only_freezes_dtype_for_key_and_build(monkeypatch, tmp_path, family):
    policy = {"cpu": torch.float32, "cuda:0": torch.float16}
    devices = []

    def choose(device):
        devices.append(device.type)
        return policy[str(device)]

    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(choose))
    path = tmp_path / f"{family}.gguf"
    if family == "ernie":
        write_ernie_image_gguf(path)
        monkeypatch.setattr(ernie_image, "ERNIE_IMAGE_TRANSFORMER_CONFIG", TINY_CONFIG)
        cls, config_cls = ErnieImageGGUFModel, Main_GGUF_ErnieImage_Config
        packed_name = "layers.0.mlp.gate_proj.weight"
    else:
        write_sd3_gguf(path)
        cls, config_cls = SD3GGUFModel, Main_GGUF_SD3_Config
        packed_name = "transformer_blocks.0.attn.to_q.weight"
    config = config_cls.model_construct(key=family, name=family, path=str(path)).model_copy(update={"cpu_only": True})
    store = SharedCpuWeightsStore()
    loader = _loader(cls, store, RamBudget(10**9, store), "cuda:0", tmp_path)
    try:
        key = loader._get_cache_key(config, SubModelType.Transformer)
        policy["cpu"] = torch.float16
        model = _load(loader, config, SubModelType.Transformer)
        packed = model.get_parameter(packed_name).data
        assert key.endswith("full-float32")
        assert isinstance(packed, GGMLTensor) and packed.compute_dtype == torch.float32
        assert devices == ["cpu"]
    finally:
        loader._ram_cache.shutdown()
