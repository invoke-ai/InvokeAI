"""Real cross-cache storage adoption; device labels select policy, all tensors stay on CPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.main import Main_Diffusers_FLUX_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.taxonomy import Krea2VariantType, MistralVariantType, SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

FP8 = torch.float8_e4m3fn


class _TinyModel(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)


class _LayerwiseLoader(ModelLoader):
    builds = 0

    def get_size_fs(self, *_args):
        return 64

    def _load_model(self, config, submodel_type=None):
        self.builds += 1
        model = _TinyModel()
        model.load_state_dict({"proj.weight": torch.ones(4, 4)}, assign=True)
        return self._apply_fp8_layerwise_casting(model, config, submodel_type)


def _loader(cls, store, budget, device, tmp_path):
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
    loader = object.__new__(cls)
    loader._ram_cache = cache
    loader._torch_device = torch.device(device)
    loader._torch_dtype = torch.float32
    loader._logger = InvokeAILogger.get_logger("test")
    loader._app_config = MagicMock(log_memory_usage=False, models_path=tmp_path)
    return loader


def _load(loader, config, subtype):
    # The legacy loaders choose the session device during construction. Match that session to the
    # simulated loader, so a red result measures cross-cache adoption, not an incorrect first build.
    with patch.object(TorchDevice, "choose_torch_device", return_value=loader._torch_device):
        record, claim = loader._load_and_cache(config, subtype)
    if claim is not None:
        claim.release()
    return record.cached_model.model


def _config():
    return Main_Diffusers_FLUX_Config.model_construct(
        key="model", name="tiny", path="tiny", default_settings=MainModelDefaultSettings(fp8_storage=True)
    )


@pytest.mark.parametrize("capable_first", [True, False])
def test_layerwise_device_caches_never_adopt_an_incompatible_storage_format(monkeypatch, tmp_path, capable_first):
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda device, _logger: device.index == 0)
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**9, shared_store=store)
    capable = _loader(_LayerwiseLoader, store, budget, "cuda:0", tmp_path)
    incapable = _loader(_LayerwiseLoader, store, budget, "cuda:1", tmp_path)
    try:
        loaders = [capable, incapable] if capable_first else [incapable, capable]
        models = {loader: _load(loader, _config(), SubModelType.Transformer) for loader in loaders}
        assert models[capable].proj.weight.dtype == FP8
        assert models[incapable].proj.weight.dtype == torch.float32
        assert models[capable].proj.weight.data_ptr() != models[incapable].proj.weight.data_ptr()
        assert capable.builds == incapable.builds == 1
        assert budget.total_in_use() == 16 + 64
    finally:
        capable._ram_cache.shutdown()
        incapable._ram_cache.shutdown()


def test_compatible_layerwise_devices_adopt_one_canonical_cpu_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda _device, _logger: True)
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**9, shared_store=store)
    first = _loader(_LayerwiseLoader, store, budget, "cuda:0", tmp_path)
    second = _loader(_LayerwiseLoader, store, budget, "cuda:1", tmp_path)
    try:
        a = _load(first, _config(), SubModelType.Transformer)
        b = _load(second, _config(), SubModelType.Transformer)
        assert a is not b
        assert a.proj.weight.dtype == b.proj.weight.dtype == FP8
        assert a.proj.weight.data_ptr() == b.proj.weight.data_ptr()
        assert first.builds == 1 and second.builds == 0
        assert budget.total_in_use() == 16
    finally:
        first._ram_cache.shutdown()
        second._ram_cache.shutdown()


def test_layerwise_key_and_build_use_one_capability_answer(monkeypatch, tmp_path):
    answers = iter([False, True])
    calls = []

    def supports(device, _logger):
        calls.append(device)
        return next(answers)

    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", supports)
    store = SharedCpuWeightsStore()
    loader = _loader(_LayerwiseLoader, store, RamBudget(10**9, store), "cuda:0", tmp_path)
    config = _config()
    try:
        loader._cache_key_variant(config, SubModelType.Transformer)
        model = _load(loader, config, SubModelType.Transformer)
        assert model.proj.weight.dtype == torch.float32
        assert len(calls) == 1
    finally:
        loader._ram_cache.shutdown()


@pytest.fixture(params=["mistral", "qwen", "krea", "ideogram"])
def checkpoint_case(request, monkeypatch):
    return _checkpoint_case(request.param, monkeypatch)


def _checkpoint_case(family, monkeypatch):
    """Replace large model geometry and checkpoint I/O; run the real quantization and loader bodies."""
    import diffusers
    import safetensors.torch

    from invokeai.backend.ideogram4 import modeling_ideogram4
    from invokeai.backend.model_manager.configs.main import (
        Main_Checkpoint_Ideogram4_Config,
        Main_Checkpoint_Krea2_Config,
        Main_Checkpoint_QwenImage_Config,
    )
    from invokeai.backend.model_manager.configs.mistral_encoder import MistralEncoder_Checkpoint_Config
    from invokeai.backend.model_manager.load.model_loaders import ideogram4, krea2, mistral_encoder, qwen_image

    cases = {
        "mistral": (mistral_encoder, mistral_encoder.MistralEncoderCheckpointLoader, MistralEncoder_Checkpoint_Config),
        "qwen": (qwen_image, qwen_image.QwenImageCheckpointModel, Main_Checkpoint_QwenImage_Config),
        "krea": (krea2, krea2.Krea2CheckpointModel, Main_Checkpoint_Krea2_Config),
        "ideogram": (ideogram4, ideogram4.Ideogram4CheckpointModel, Main_Checkpoint_Ideogram4_Config),
    }
    module, cls, config_class = cases[family]
    subtype = SubModelType.TextEncoder if family == "mistral" else SubModelType.Transformer
    prefix = "model." if family == "mistral" else ""
    source = {prefix + "proj.weight": torch.ones(4, 4).to(FP8), prefix + "proj.weight_scale": torch.tensor(0.5)}
    reads = []

    def read(path):
        reads.append(str(path))
        return {key: tensor.clone() for key, tensor in source.items()}

    monkeypatch.setattr(safetensors.torch, "load_file", read)
    if family == "ideogram":
        monkeypatch.setattr(ideogram4, "load_file", read)
        monkeypatch.setattr(modeling_ideogram4, "Ideogram4Transformer", _TinyModel)
    elif family == "qwen":
        monkeypatch.setattr(diffusers, "QwenImageTransformer2DModel", _TinyModel)
        monkeypatch.setattr(qwen_image, "_build_qwen_image_transformer_config", lambda *_a, **_kw: {})
    elif family == "krea":
        # Model construction is the same boundary as test_krea2_loader_boundaries.py.
        monkeypatch.setattr(diffusers, "Krea2Transformer2DModel", _TinyModel, raising=False)
    else:
        monkeypatch.setattr(mistral_encoder, "MistralModel", _TinyModel)
        monkeypatch.setattr(
            mistral_encoder,
            "_build_mistral_config",
            lambda *_a, **_kw: SimpleNamespace(
                num_hidden_layers=1, hidden_size=4, num_attention_heads=1, num_key_value_heads=1, intermediate_size=4
            ),
        )
    monkeypatch.setattr(module, "read_safetensors_metadata", lambda *_a: {})
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda device: device.index == 0)
    for host in (mistral_encoder, qwen_image):
        if hasattr(host, "should_keep_fp8_weights"):
            monkeypatch.setattr(host, "should_keep_fp8_weights", lambda device: device.index == 0)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: False)
    if hasattr(mistral_encoder, "_device_supports_fp8_storage"):
        monkeypatch.setattr(mistral_encoder, "_device_supports_fp8_storage", lambda *_a: False)

    def config(**extra):
        if family == "mistral":
            extra.setdefault("variant", MistralVariantType.Mistral24B)
        if family == "krea":
            extra.setdefault("variant", Krea2VariantType.Turbo)
        return config_class.model_construct(key="checkpoint", name="tiny", path="tiny.safetensors", **extra)

    return cls, config, subtype, reads


@pytest.mark.parametrize("capable_first", [True, False])
def test_checkpoint_caches_preserve_fp8_or_folded_storage(checkpoint_case, monkeypatch, tmp_path, capable_first):
    cls, config, subtype, reads = checkpoint_case
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    capable = _loader(cls, store, budget, "cuda:0", tmp_path)
    incapable = _loader(cls, store, budget, "cuda:1", tmp_path)
    try:
        loaders = [capable, incapable] if capable_first else [incapable, capable]
        first = _load(loaders[0], config(), subtype)
        assert first.proj.weight.dtype == (FP8 if capable_first else torch.float32)
        assert len(reads) == 1
        models = {loaders[0]: first, loaders[1]: _load(loaders[1], config(), subtype)}
        assert models[capable].proj.weight.dtype == FP8
        assert models[incapable].proj.weight.dtype == torch.float32
        torch.testing.assert_close(models[incapable].proj.weight, torch.full((4, 4), 0.5))
        assert models[capable].proj.weight.data_ptr() != models[incapable].proj.weight.data_ptr()
        assert len(reads) == 2
        # A third cache with matching policy adopts the first device's canonical storage without reading again.
        peer = _loader(cls, store, budget, "cuda:0", tmp_path)
        try:
            adopted = _load(peer, config(), subtype)
            assert adopted.proj.weight.data_ptr() == models[capable].proj.weight.data_ptr()
            assert len(reads) == 2
        finally:
            peer._ram_cache.shutdown()
    finally:
        capable._ram_cache.shutdown()
        incapable._ram_cache.shutdown()


@pytest.mark.parametrize("reverse", [True, False])
def test_checkpoint_caches_do_not_adopt_a_different_folded_dtype(checkpoint_case, monkeypatch, tmp_path, reverse):
    cls, config, subtype, reads = checkpoint_case
    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder, qwen_image

    for module in (load_default, mistral_encoder, qwen_image):
        if hasattr(module, "should_keep_fp8_weights"):
            monkeypatch.setattr(module, "should_keep_fp8_weights", lambda _d: False)
    monkeypatch.setattr(
        TorchDevice,
        "choose_bfloat16_safe_dtype",
        staticmethod(lambda d: torch.float16 if d.index == 0 else torch.float32),
    )
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    half = _loader(cls, store, budget, "cuda:0", tmp_path)
    full = _loader(cls, store, budget, "cuda:1", tmp_path)
    try:
        loaders = [full, half] if reverse else [half, full]
        first = _load(loaders[0], config(), subtype)
        assert first.proj.weight.dtype == (torch.float32 if reverse else torch.float16)
        assert len(reads) == 1
        models = {loaders[0]: first, loaders[1]: _load(loaders[1], config(), subtype)}
        assert models[half].proj.weight.dtype == torch.float16
        assert models[full].proj.weight.dtype == torch.float32
        assert models[half].proj.weight.data_ptr() != models[full].proj.weight.data_ptr()
        assert len(reads) == 2
    finally:
        half._ram_cache.shutdown()
        full._ram_cache.shutdown()


def test_checkpoint_key_freezes_the_build_capability_and_dtype(checkpoint_case, monkeypatch, tmp_path):
    cls, config_factory, subtype, _reads = checkpoint_case
    from invokeai.backend.model_manager.load.model_loaders import mistral_encoder, qwen_image

    state = {"keep": False, "dtype": torch.float32}
    calls = []

    def keep(_device):
        calls.append(True)
        return state["keep"]

    for module in (load_default, mistral_encoder, qwen_image):
        if hasattr(module, "should_keep_fp8_weights"):
            monkeypatch.setattr(module, "should_keep_fp8_weights", keep)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: state["dtype"]))
    store = SharedCpuWeightsStore()
    loader = _loader(cls, store, RamBudget(10**9, store), "cuda:0", tmp_path)
    config = config_factory()
    try:
        loader._cache_key_variant(config, subtype)
        state.update(keep=True, dtype=torch.float16)
        model = _load(loader, config, subtype)
        assert model.proj.weight.dtype == torch.float32
        assert len(calls) == 1
    finally:
        loader._ram_cache.shutdown()


def test_cpu_only_mistral_shares_with_cpu_instead_of_the_loaders_gpu(monkeypatch, tmp_path):
    cls, config, subtype, reads = _checkpoint_case("mistral", monkeypatch)
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    cpu = _loader(cls, store, budget, "cpu", tmp_path)
    cpu_only = _loader(cls, store, budget, "cuda:0", tmp_path)
    try:
        a = _load(cpu, config(), subtype)
        b = _load(cpu_only, config(cpu_only=True), subtype)
        assert a.proj.weight.dtype == b.proj.weight.dtype == torch.float32
        assert a.proj.weight.data_ptr() == b.proj.weight.data_ptr()
        assert len(reads) == 1
        # The original GPU policy still needs its own fp8 canonical copy.
        gpu = _loader(cls, store, budget, "cuda:0", tmp_path)
        try:
            c = _load(gpu, config(), subtype)
            assert c.proj.weight.dtype == FP8
            assert c.proj.weight.data_ptr() != a.proj.weight.data_ptr()
            assert len(reads) == 2
        finally:
            gpu._ram_cache.shutdown()
    finally:
        cpu._ram_cache.shutdown()
        cpu_only._ram_cache.shutdown()


def test_single_file_prefetch_uses_the_same_variant_as_a_later_load(monkeypatch, tmp_path):
    from invokeai.backend.model_manager.configs.main import Main_Checkpoint_SD1_Config
    from invokeai.backend.model_manager.load.model_loaders.stable_diffusion import (
        StableDiffusionDiffusersModel,
        StableDiffusionPipeline,
    )
    from invokeai.backend.model_manager.taxonomy import ModelVariantType

    calls = []

    def pipeline(*_a, **_kw):
        calls.append(True)
        parts = {}
        for name in ("unet", "text_encoder", "vae"):
            model = _TinyModel()
            model.load_state_dict({"proj.weight": torch.ones(4, 4)}, assign=True)
            parts[name] = model
        return SimpleNamespace(**parts)

    monkeypatch.setattr(StableDiffusionPipeline, "from_single_file", pipeline)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    store = SharedCpuWeightsStore()
    loader = _loader(StableDiffusionDiffusersModel, store, RamBudget(10**9, store), "cuda:0", tmp_path)
    config = Main_Checkpoint_SD1_Config.model_construct(
        key="pipeline",
        name="tiny",
        path="tiny.safetensors",
        variant=ModelVariantType.Normal,
        default_settings=MainModelDefaultSettings(fp8_storage=True),
    )
    try:
        _load(loader, config, SubModelType.TextEncoder)
        model = _load(loader, config, SubModelType.UNet)
        assert model.proj.weight.dtype == FP8
        assert len(calls) == 1
        assert store.peek("pipeline:unet") is None
    finally:
        loader._ram_cache.shutdown()
