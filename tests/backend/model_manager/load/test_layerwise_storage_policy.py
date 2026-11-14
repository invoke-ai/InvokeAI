"""Exercise storage policy through public loaders, real admission, shells and CPU tensors."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import FP8_COMPUTE_DTYPE_ATTR
from tests.backend.model_manager.load.test_fp8_shared_weight_variants import _loader as device_loader


class _TinyTransformer(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)
        self.rope = torch.nn.Module()
        self.rope.register_buffer("bands", torch.ones(2))


def _public_load(loader, config, subtype=None):
    with patch.object(TorchDevice, "choose_torch_device", return_value=loader._torch_device):
        loaded = loader.load_model(config, subtype)
        with loaded.model_in_ram() as model:
            return model


def _constructor_case(family, monkeypatch, tmp_path):
    """Only geometry/metadata and heavy I/O are replaced; the loader constructor path is real."""
    import diffusers
    import safetensors.torch

    from invokeai.backend.model_manager.configs import main
    from invokeai.backend.model_manager.load.model_loaders import (
        cogview4,
        ernie_image,
        flux,
        krea2,
        moss_image,
        qwen_image,
        qwen_image_2_1,
        wan,
        z_image,
    )
    from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader

    cases = {
        "z": (z_image.ZImageDiffusersModel, main.Main_Diffusers_ZImage_Config),
        "krea": (krea2.Krea2DiffusersModel, main.Main_Diffusers_Krea2_Config),
        "moss": (moss_image.MossImageCheckpointModel, main.Main_Checkpoint_MossImage_Config),
        "moss-folder": (moss_image.MossImageDiffusersModel, main.Main_Diffusers_MossImage_Config),
        "ernie": (ernie_image.ErnieImageCheckpointModel, main.Main_Checkpoint_ErnieImage_Config),
        "flux": (flux.FluxDiffusersModel, main.Main_Diffusers_FLUX_Config),
        "flux2": (flux.Flux2DiffusersModel, main.Main_Diffusers_Flux2_Config),
        "qwen": (qwen_image.QwenImageDiffusersModel, main.Main_Diffusers_QwenImage_Config),
        "cogview": (cogview4.CogView4DiffusersModel, main.Main_Diffusers_CogView4_Config),
        "wan": (wan.WanDiffusersModel, main.Main_Diffusers_Wan_Config),
        "ernie-folder": (ernie_image.ErnieImageDiffusersModel, main.Main_Diffusers_ErnieImage_Config),
        "qwen21": (qwen_image_2_1.QwenImage21DiffusersModel, main.Main_Diffusers_QwenImage21_Config),
        "generic": (GenericDiffusersLoader, main.Main_Diffusers_FLUX_Config),
    }
    cls, config_cls = cases[family]
    reads = []
    source = {key: tensor.detach().clone() for key, tensor in _TinyTransformer().state_dict().items()}

    def read(*_args):
        reads.append(True)
        return {key: tensor.clone() for key, tensor in source.items()}

    class TinyPretrained:
        @staticmethod
        def from_pretrained(_path, *, torch_dtype=None, dtype=None, **_kwargs):
            compute_dtype = torch_dtype if torch_dtype is not None else dtype
            assert compute_dtype is not None
            model = _TinyTransformer().to(compute_dtype)
            model.load_state_dict({k: v.to(compute_dtype) for k, v in read().items()}, assign=True)
            return model

    if family in ("moss", "moss-folder"):
        monkeypatch.setattr(moss_image, "MOSSImageTransformer", _TinyTransformer)
        monkeypatch.setattr(moss_image, "moss_image_transformer_config", lambda _sd: {})
        monkeypatch.setattr(moss_image, "load_file", read)
        monkeypatch.setattr(moss_image, "_read_component_state_dict", read)
    elif family == "ernie":
        monkeypatch.setattr(diffusers, "ErnieImageTransformer2DModel", _TinyTransformer)
        monkeypatch.setattr(safetensors.torch, "load_file", read)
    else:
        monkeypatch.setattr(cls, "get_hf_load_class", lambda *_a: TinyPretrained)

    model_path = tmp_path / family
    model_path.mkdir()
    if family == "moss-folder":
        (model_path / "transformer").mkdir()
        (model_path / "transformer" / "config.json").write_text("{}")

    def config():
        return config_cls.model_construct(
            key="same-model",
            name="tiny",
            path=str(model_path),
            repo_variant=None,
            default_settings=MainModelDefaultSettings(fp8_storage=True),
        )

    return cls, config, reads


@pytest.mark.parametrize("family", ["moss", "ernie"])
@pytest.mark.parametrize("changing_first", [True, False])
def test_default_transformer_alias_freezes_capability_before_public_adoption(
    family, changing_first, monkeypatch, tmp_path
):
    cls, config, reads = _constructor_case(family, monkeypatch, tmp_path)
    state = {"supports": False}
    calls = []

    def supports(device, _logger):
        calls.append(device)
        return state["supports"] if device.index == 0 else False

    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", supports)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    changing = device_loader(cls, store, budget, "cuda:0", tmp_path)
    unsupported = device_loader(cls, store, budget, "cuda:1", tmp_path)

    @contextmanager
    def change_after_key(_config):
        state["supports"] = True
        yield lambda: True

    changing.config_check = change_after_key
    loaders = [changing, unsupported] if changing_first else [unsupported, changing]
    try:
        first = _public_load(loaders[0], config())
        second = _public_load(loaders[1], config())
        assert first.proj.weight.dtype == second.proj.weight.dtype == torch.float32
        assert getattr(first, FP8_COMPUTE_DTYPE_ATTR, None) is None
        assert getattr(second, FP8_COMPUTE_DTYPE_ATTR, None) is None
        assert first.proj.weight.data_ptr() == second.proj.weight.data_ptr()
        assert calls.count(torch.device("cuda:0")) == 1
        assert len(reads) == 1
        # Explicit Transformer is the same public component as the default, including across caches.
        state["supports"] = False
        peer = device_loader(cls, store, budget, "cuda:2", tmp_path)
        try:
            explicit = _public_load(peer, config(), SubModelType.Transformer)
            assert explicit.proj.weight.data_ptr() == first.proj.weight.data_ptr()
            assert len(reads) == 1
        finally:
            peer._ram_cache.shutdown()
    finally:
        changing._ram_cache.shutdown()
        unsupported._ram_cache.shutdown()


@pytest.mark.parametrize("family", ["z", "krea", "moss", "moss-folder"])
@pytest.mark.parametrize("reverse", [False, True])
def test_equal_global_dtype_does_not_share_different_constructor_dtypes(family, reverse, monkeypatch, tmp_path):
    cls, config, reads = _constructor_case(family, monkeypatch, tmp_path)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    monkeypatch.setattr(
        TorchDevice,
        "choose_bfloat16_safe_dtype",
        staticmethod(lambda d: torch.float16 if d.index == 0 else torch.float32),
    )
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    half = device_loader(cls, store, budget, "cuda:0", tmp_path)
    full = device_loader(cls, store, budget, "cuda:1", tmp_path)
    try:
        loaders = [full, half] if reverse else [half, full]
        models = {loader: _public_load(loader, config(), SubModelType.Transformer) for loader in loaders}
        for loader, dtype in ((half, torch.float16), (full, torch.float32)):
            assert models[loader].norm.weight.dtype == dtype
            assert getattr(models[loader], FP8_COMPUTE_DTYPE_ATTR) == dtype
        assert models[half].proj.weight.data_ptr() != models[full].proj.weight.data_ptr()
        assert len(reads) == 2
    finally:
        half._ram_cache.shutdown()
        full._ram_cache.shutdown()


def test_flux_fixed_bf16_shares_despite_different_global_dtypes(monkeypatch, tmp_path):
    cls, config, reads = _constructor_case("flux", monkeypatch, tmp_path)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    a = device_loader(cls, store, budget, "cuda:0", tmp_path)
    b = device_loader(cls, store, budget, "cuda:1", tmp_path)
    a._torch_dtype, b._torch_dtype = torch.float16, torch.float32
    try:
        first = _public_load(a, config(), SubModelType.Transformer)
        second = _public_load(b, config(), SubModelType.Transformer)
        assert first.norm.weight.dtype == second.norm.weight.dtype == torch.bfloat16
        assert getattr(first, FP8_COMPUTE_DTYPE_ATTR) == getattr(second, FP8_COMPUTE_DTYPE_ATTR) == torch.bfloat16
        assert first.proj.weight.data_ptr() == second.proj.weight.data_ptr()
        assert len(reads) == 1
    finally:
        a._ram_cache.shutdown()
        b._ram_cache.shutdown()


@pytest.mark.parametrize("family", ["flux2", "qwen", "cogview", "wan"])
def test_fixed_bf16_constructors_share_one_canonical_copy_across_global_precisions(family, monkeypatch, tmp_path):
    cls, config, reads = _constructor_case(family, monkeypatch, tmp_path)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    first_loader = device_loader(cls, store, budget, "cuda:0", tmp_path)
    second_loader = device_loader(cls, store, budget, "cuda:1", tmp_path)
    peer_loader = device_loader(cls, store, budget, "cuda:2", tmp_path)
    first_loader._torch_dtype = peer_loader._torch_dtype = torch.float16
    second_loader._torch_dtype = torch.float32
    try:
        first = _public_load(first_loader, config(), SubModelType.Transformer)
        second = _public_load(second_loader, config(), SubModelType.Transformer)
        # Both constructors already force BF16 on V2: only identity/sharing should fail there.
        assert first.norm.weight.dtype == second.norm.weight.dtype == torch.bfloat16
        assert getattr(first, FP8_COMPUTE_DTYPE_ATTR) == getattr(second, FP8_COMPUTE_DTYPE_ATTR) == torch.bfloat16
        assert first.proj.weight.data_ptr() == second.proj.weight.data_ptr()
        assert len(reads) == 1
        peer = _public_load(peer_loader, config(), SubModelType.Transformer)
        assert peer.proj.weight.data_ptr() == first.proj.weight.data_ptr()
        assert len(reads) == 1
    finally:
        first_loader._ram_cache.shutdown()
        second_loader._ram_cache.shutdown()
        peer_loader._ram_cache.shutdown()


@pytest.mark.parametrize("family", ["ernie-folder", "qwen21"])
@pytest.mark.parametrize("reverse", [False, True])
def test_additional_bfloat_safe_constructors_partition_actual_dtypes(family, reverse, monkeypatch, tmp_path):
    cls, config, reads = _constructor_case(family, monkeypatch, tmp_path)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    monkeypatch.setattr(
        TorchDevice,
        "choose_bfloat16_safe_dtype",
        staticmethod(lambda d: torch.float16 if d.index == 0 else torch.float32),
    )
    store = SharedCpuWeightsStore()
    budget = RamBudget(10**9, store)
    half = device_loader(cls, store, budget, "cuda:0", tmp_path)
    full = device_loader(cls, store, budget, "cuda:1", tmp_path)
    try:
        loaders = [full, half] if reverse else [half, full]
        first = _public_load(loaders[0], config(), SubModelType.Transformer)
        assert first.norm.weight.dtype == (torch.float32 if reverse else torch.float16)
        assert len(reads) == 1
        models = {loaders[0]: first, loaders[1]: _public_load(loaders[1], config(), SubModelType.Transformer)}
        for loader, dtype in ((half, torch.float16), (full, torch.float32)):
            assert models[loader].norm.weight.dtype == dtype
            assert getattr(models[loader], FP8_COMPUTE_DTYPE_ATTR) == dtype
        assert models[half].proj.weight.data_ptr() != models[full].proj.weight.data_ptr()
        assert len(reads) == 2
    finally:
        half._ram_cache.shutdown()
        full._ram_cache.shutdown()


def test_generic_loader_keeps_its_configured_constructor_dtype(monkeypatch, tmp_path):
    cls, config, _reads = _constructor_case("generic", monkeypatch, tmp_path)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: torch.float32))
    store = SharedCpuWeightsStore()
    loader = device_loader(cls, store, RamBudget(10**9, store), "cuda:0", tmp_path)
    loader._torch_dtype = torch.float16
    try:
        model = _public_load(loader, config())
        assert model.norm.weight.dtype == torch.float16
        assert getattr(model, FP8_COMPUTE_DTYPE_ATTR) == torch.float16
    finally:
        loader._ram_cache.shutdown()


@pytest.mark.parametrize("family", ["z", "krea", "moss", "moss-folder"])
def test_constructor_reuses_the_dtype_selected_before_the_load_lock(family, monkeypatch, tmp_path):
    cls, config, _reads = _constructor_case(family, monkeypatch, tmp_path)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_a: True)
    state = {"dtype": torch.float32}
    choices = []

    def choose(_device):
        choices.append(state["dtype"])
        return state["dtype"]

    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    store = SharedCpuWeightsStore()
    loader = device_loader(cls, store, RamBudget(10**9, store), "cuda:0", tmp_path)

    @contextmanager
    def change_after_key(_config):
        state["dtype"] = torch.float16
        yield lambda: True

    loader.config_check = change_after_key
    try:
        model = _public_load(loader, config(), SubModelType.Transformer)
        assert model.norm.weight.dtype == torch.float32
        assert getattr(model, FP8_COMPUTE_DTYPE_ATTR) == torch.float32
        assert choices == [torch.float32]
    finally:
        loader._ram_cache.shutdown()


def test_a_pipeline_none_still_loads_the_whole_pipeline(monkeypatch, tmp_path):
    from invokeai.backend.model_manager.configs.main import Main_Checkpoint_SD1_Config
    from invokeai.backend.model_manager.load.model_loaders.stable_diffusion import (
        StableDiffusionDiffusersModel,
        StableDiffusionPipeline,
    )
    from invokeai.backend.model_manager.taxonomy import ModelVariantType

    pipeline = SimpleNamespace(unet=_TinyTransformer(), text_encoder=_TinyTransformer())
    monkeypatch.setattr(StableDiffusionPipeline, "from_single_file", lambda *_a, **_kw: pipeline)
    path = tmp_path / "pipeline.safetensors"
    path.touch()
    config = Main_Checkpoint_SD1_Config.model_construct(
        key="pipeline",
        name="pipeline",
        path=str(path),
        variant=ModelVariantType.Normal,
    )
    store = SharedCpuWeightsStore()
    loader = device_loader(StableDiffusionDiffusersModel, store, RamBudget(10**9, store), "cpu", tmp_path)
    try:
        assert _public_load(loader, config) is pipeline
    finally:
        loader._ram_cache.shutdown()
