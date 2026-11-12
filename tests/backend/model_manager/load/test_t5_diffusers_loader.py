"""Loading a T5 encoder from a Diffusers folder (``text_encoder_2/``), the layout FLUX.1 and SD3 ship it in.

The folder stores bf16, as the real ones do. Which dtype the loader holds it in depends on the device the
encoder runs on: float32 on the CPU (bf16 GEMM faults on part of the Windows fleet), otherwise whatever
``choose_bfloat16_safe_dtype`` picks, which these tests patch to float16 so a GPU decision shows.

Forwards run in float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from transformers import T5Config, T5EncoderModel

from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_T5Encoder_Config
from invokeai.backend.model_manager.load.load_default import put_in_eval_mode
from invokeai.backend.model_manager.load.model_loaders.flux import T5EncoderCheckpointModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


def _write_bf16_folder(root: Path, max_shard_size: str | None = None) -> T5EncoderModel:
    """A tiny T5 v1.1 saved in bf16 under ``root/text_encoder_2``, split into shards of at most ``max_shard_size``
    when given; returns the model as stored (bf16 values)."""
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_heads=4,
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        dense_act_fn="gelu_new",
        use_cache=False,
    )
    model = T5EncoderModel(config).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.5)
    shard_kwargs = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    model.to(torch.bfloat16).save_pretrained(root / "text_encoder_2", **shard_kwargs)
    return model


def _config(path: Path, cpu_only: bool | None = None) -> T5Encoder_T5Encoder_Config:
    return T5Encoder_T5Encoder_Config.model_construct(
        key="t5", path=str(path), name="t5", base=BaseModelType.Any, type=ModelType.T5Encoder, cpu_only=cpu_only
    )


@pytest.fixture
def loader() -> T5EncoderCheckpointModel:
    loader = object.__new__(T5EncoderCheckpointModel)
    loader._ram_cache = MagicMock()
    loader._logger = InvokeAILogger.get_logger("test")
    loader._torch_device = torch.device("cpu")
    return loader


def _load_encoder(loader: T5EncoderCheckpointModel, config: T5Encoder_T5Encoder_Config) -> T5EncoderModel:
    return put_in_eval_mode(loader._load_model(config, SubModelType.TextEncoder2))


def test_a_cpu_load_holds_float32_and_encodes_like_the_stored_weights(
    loader: T5EncoderCheckpointModel, tmp_path: Path
) -> None:
    stored = _write_bf16_folder(tmp_path)

    model = _load_encoder(loader, _config(tmp_path))

    assert all(p.dtype == torch.float32 for p in model.parameters())
    # The stored bf16 values widen exactly, so a float32 forward over them is the expected output.
    expected = T5EncoderModel(stored.config).eval()
    expected.load_state_dict({k: v.float() for k, v in stored.state_dict().items()}, strict=False)
    input_ids = torch.tensor([[3, 7, 1, 30, 12, 0]])
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=input_ids).last_hidden_state, expected(input_ids=input_ids).last_hidden_state
        )


def test_a_cpu_only_t5_on_a_gpu_box_holds_float32(
    loader: T5EncoderCheckpointModel, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `cpu_only` runs the encoder on the CPU while the loader's own device is the GPU: the dtype follows the
    # execution device, not `_torch_device`.
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    _write_bf16_folder(tmp_path)
    loader._torch_device = torch.device("cuda:0")
    config = _config(tmp_path, cpu_only=True)

    assert loader._cache_key_variant(config, SubModelType.TextEncoder2) == "float32"
    model = _load_encoder(loader, config)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_a_gpu_box_holds_the_dtype_the_device_picks_and_keys_the_cache_by_it(
    loader: T5EncoderCheckpointModel, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    _write_bf16_folder(tmp_path)
    loader._torch_device = torch.device("cuda:0")
    config = _config(tmp_path)

    assert loader._cache_key_variant(config, SubModelType.TextEncoder2) == "float16"
    assert loader._cache_key_variant(config, SubModelType.Tokenizer2) == ""
    model = _load_encoder(loader, config)
    # transformers keeps T5's `wo` in float32 under float16 (its activations overflow); the rest follows the device.
    dtypes = {n: p.dtype for n, p in model.named_parameters() if not n.endswith("DenseReluDense.wo.weight")}
    assert set(dtypes.values()) == {torch.float16}


def _device_loader(store, budget, device: str, tmp_path: Path) -> T5EncoderCheckpointModel:
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
    loader = object.__new__(T5EncoderCheckpointModel)
    loader._app_config = MagicMock(log_memory_usage=False, models_path=tmp_path)
    loader._logger = InvokeAILogger.get_logger("test")
    loader._ram_cache = cache
    loader._torch_device = torch.device(device)
    return loader


@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu_loads_first", "cuda_loads_first"])
def test_device_caches_do_not_share_weights_of_different_dtypes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_first: bool
) -> None:
    # CPU holds float32 and a CUDA device without bf16 holds float16: the shared CPU store must not hand one
    # device the other's dtype.
    from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
    from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore

    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    _write_bf16_folder(tmp_path)
    config = _config(Path("."))
    config.path = "."  # relative to the loaders' `models_path`
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)

    loaders = {}  # a device's cache entry holds the shared weights only while its cache lives

    def weight_dtype(device: str) -> torch.dtype:
        loaders[device] = _device_loader(store, budget, device, tmp_path)
        record, _claim = loaders[device]._load_and_cache(config, SubModelType.TextEncoder2)
        return record.cached_model.model.encoder.block[0].layer[0].SelfAttention.q.weight.dtype

    dtypes = {}
    for name, device in [("cpu", "cpu"), ("cuda", "cuda:0")] if cpu_first else [("cuda", "cuda:0"), ("cpu", "cpu")]:
        dtypes[name] = weight_dtype(device)

    assert dtypes == {"cpu": torch.float32, "cuda": torch.float16}


@pytest.mark.parametrize("sharded", [False, True], ids=["single_file", "sharded"])
def test_a_cold_cpu_load_makes_room_for_the_float32_encoder_not_the_bf16_folder(tmp_path: Path, sharded: bool) -> None:
    # The folder stores bf16 but the CPU builds float32, twice the folder: reserving the folder's size would let a
    # cold load overshoot the RAM budget by about a file. Real FLUX.1/SD3 folders shard the encoder, so the size has
    # to come from every shard the index names.
    from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
    from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore

    _write_bf16_folder(tmp_path, max_shard_size="4KB" if sharded else None)
    if sharded:
        index = json.loads((tmp_path / "text_encoder_2" / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert len(set(index["weight_map"].values())) > 1
    config = _config(Path("."))
    config.path = "."  # relative to the loader's `models_path`
    store = SharedCpuWeightsStore()
    loader = _device_loader(store, RamBudget(max_bytes=10**12, shared_store=store), "cpu", tmp_path)
    reservations: list[int] = []
    make_room = loader._ram_cache.make_room

    def recording_make_room(needed: int):
        reservations.append(needed)
        return make_room(needed)

    loader._ram_cache.make_room = recording_make_room

    record, _claim = loader._load_and_cache(config, SubModelType.TextEncoder2)

    model = record.cached_model.model
    tensors = {t.untyped_storage().data_ptr(): t for t in [*model.parameters(), *model.buffers()]}
    held = sum(t.nbytes for t in tensors.values())
    assert max(reservations) >= held
