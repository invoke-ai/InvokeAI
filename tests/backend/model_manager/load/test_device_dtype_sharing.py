"""Device policies are simulated; model weights and shared storage remain on CPU."""

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Wan_Config,
    Main_Diffusers_Flux3_Config,
    Main_Diffusers_MossImage_Config,
    Main_Diffusers_QwenImage21_Config,
)
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_loaders.ernie_image import ErnieImageDiffusersModel
from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3DiffusersModel
from invokeai.backend.model_manager.load.model_loaders.krea2 import Krea2DiffusersModel
from invokeai.backend.model_manager.load.model_loaders.ltx2 import LTX2FolderModel
from invokeai.backend.model_manager.load.model_loaders.moss_image import MossImageDiffusersModel
from invokeai.backend.model_manager.load.model_loaders.qwen_image_2_1 import QwenImage21DiffusersModel
from invokeai.backend.model_manager.load.model_loaders.wan import WanCheckpointModel, WanDiffusersModel
from invokeai.backend.model_manager.taxonomy import SubModelType, WanVariantType
from invokeai.backend.util.devices import TorchDevice
from tests.backend.model_manager.load.test_fp8_shared_weight_variants import _load, _loader
from tests.backend.model_manager.load.test_wan_checkpoint_loader import _tiny_model


@pytest.mark.parametrize("reverse", [False, True])
def test_wan_checkpoint_distinct_dtypes_do_not_adopt_each_other(monkeypatch, tmp_path, reverse):
    monkeypatch.setattr(
        TorchDevice,
        "choose_bfloat16_safe_dtype",
        staticmethod(lambda d: torch.float16 if d.index == 0 else torch.float32),
    )
    path = tmp_path / "wan.safetensors"
    save_file(_tiny_model().state_dict(), str(path))
    config = Main_Checkpoint_Wan_Config.model_construct(
        key="wan", name="wan", path=str(path), variant=WanVariantType.T2V_A14B
    )
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    half = _loader(WanCheckpointModel, store, budget, "cuda:0", tmp_path)
    full = _loader(WanCheckpointModel, store, budget, "cuda:1", tmp_path)
    peer = _loader(WanCheckpointModel, store, budget, "cuda:0", tmp_path)
    try:
        order = [full, half] if reverse else [half, full]
        models = {loader: _load(loader, config, SubModelType.Transformer) for loader in order}
        a = models[half].proj_out.weight
        b = models[full].proj_out.weight
        assert a.dtype == torch.float16 and b.dtype == torch.float32
        assert a.data_ptr() != b.data_ptr()
        assert half._get_cache_key(config, SubModelType.Transformer) != full._get_cache_key(
            config, SubModelType.Transformer
        )
        adopted = _load(peer, config, SubModelType.Transformer).proj_out.weight
        assert adopted.data_ptr() == a.data_ptr()
        torch.testing.assert_close(a.float(), b, rtol=1e-3, atol=1e-4)
    finally:
        for loader in (half, full, peer):
            loader._ram_cache.shutdown()


def test_component_variants_follow_actual_dtype_and_keep_dtype_independent_components_shared(monkeypatch, tmp_path):
    monkeypatch.setattr(
        TorchDevice,
        "choose_bfloat16_safe_dtype",
        staticmethod(lambda d: torch.float16 if d.index == 0 else torch.float32),
    )
    config = Main_Checkpoint_Wan_Config.model_construct(key="model", name="m", path="m")
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    loaders = []
    try:

        def keys(cls, subtype):
            a = _loader(cls, store, budget, "cuda:0", tmp_path)
            b = _loader(cls, store, budget, "cuda:1", tmp_path)
            loaders.extend((a, b))
            return a._get_cache_key(config, subtype), b._get_cache_key(config, subtype)

        assert len(set(keys(ErnieImageDiffusersModel, SubModelType.TextEncoder))) == 2
        assert len(set(keys(ErnieImageDiffusersModel, SubModelType.Tokenizer))) == 1
        assert len(set(keys(ErnieImageDiffusersModel, SubModelType.PromptEnhancerTokenizer))) == 1
        assert len(set(keys(Krea2DiffusersModel, SubModelType.Processor))) == 1
        assert len(set(keys(WanDiffusersModel, SubModelType.VAE))) == 1
        assert len(set(keys(LTX2FolderModel, SubModelType.VAE))) == 2
        assert len(set(keys(LTX2FolderModel, SubModelType.AudioVAE))) == 1
        assert len(set(keys(LTX2FolderModel, SubModelType.Vocoder))) == 1
        for cls in (Flux3DiffusersModel, MossImageDiffusersModel, QwenImage21DiffusersModel):
            # The configured float32 is fixed even when the device's preferred dtype differs.
            assert len(set(keys(cls, SubModelType.VAE))) == 1
    finally:
        for loader in loaders:
            loader._ram_cache.shutdown()


@pytest.mark.parametrize(
    ("cls", "config_cls"),
    [
        (Flux3DiffusersModel, Main_Diffusers_Flux3_Config),
        (MossImageDiffusersModel, Main_Diffusers_MossImage_Config),
        (QwenImage21DiffusersModel, Main_Diffusers_QwenImage21_Config),
    ],
)
def test_vae_build_uses_the_dtype_selected_for_its_cache_key(monkeypatch, tmp_path, cls, config_cls):
    import diffusers

    state = {"dtype": torch.float32}
    built = []
    monkeypatch.setattr(TorchDevice, "dtype_avoiding_float16", lambda *_args: state["dtype"])
    monkeypatch.setattr(
        diffusers.AutoencoderKLFlux2,
        "from_pretrained",
        lambda *_args, **kwargs: built.append(kwargs["torch_dtype"]) or torch.nn.Linear(1, 1),
    )
    store = SharedCpuWeightsStore()
    loader = _loader(cls, store, RamBudget(10**9, store), "cuda:0", tmp_path)
    config = config_cls.model_construct(key="vae", name="vae", path=str(tmp_path))
    try:
        if isinstance(loader, QwenImage21DiffusersModel):
            monkeypatch.setattr(loader, "get_hf_load_class", lambda *_args: diffusers.AutoencoderKLFlux2)
            monkeypatch.setattr(loader, "_apply_fp8_layerwise_casting", lambda model, *_args: model)
        key = loader._get_cache_key(config, SubModelType.VAE)
        state["dtype"] = torch.bfloat16
        loader._load_model(config, SubModelType.VAE)
        assert key.endswith("full-float32")
        assert built == [torch.float32]
    finally:
        loader._ram_cache.shutdown()
