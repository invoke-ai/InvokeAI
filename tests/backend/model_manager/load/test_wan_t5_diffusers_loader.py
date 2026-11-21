"""Native CPU coverage for standalone Wan UMT5 folder loading and cache admission."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from transformers import UMT5Config, UMT5EncoderModel

from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_WanT5Encoder_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache, get_model_cache_key
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_loaders.wan import WanT5EncoderLoader
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

ENCODER = SubModelType.TextEncoder


def _write_encoder(folder: Path, sharded: bool = False, **config_overrides) -> UMT5EncoderModel:
    config = {
        "vocab_size": 128,
        "d_model": 16,
        "d_kv": 3,
        "d_ff": 32,
        "num_layers": 2,
        "num_heads": 4,
        "use_cache": False,
    }
    config.update(config_overrides)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        model = UMT5EncoderModel(UMT5Config(**config)).eval().to(torch.bfloat16)
    model.save_pretrained(folder, max_shard_size="4KB" if sharded else "5GB")
    return model


def _config(path: Path, cpu_only: bool | None = None) -> WanT5Encoder_WanT5Encoder_Config:
    return WanT5Encoder_WanT5Encoder_Config.model_construct(key="wan", name="wan", path=str(path), cpu_only=cpu_only)


class _CpuHostedCache(ModelCache):
    """Expose the selected device while keeping model storage and test forwards on CPU."""

    selected_device: torch.device

    @property
    def execution_device(self) -> torch.device:
        return self.selected_device


def _loader(
    root: Path,
    device: str = "cpu",
    budget: RamBudget | None = None,
    store: SharedCpuWeightsStore | None = None,
    *,
    loader_device: str | None = None,
) -> WanT5EncoderLoader:
    logger = InvokeAILogger.get_logger("test")
    cache = _CpuHostedCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
        shared_cpu_weights=store,
        ram_budget=budget,
    )
    cache.selected_device = torch.device(device)
    loader = object.__new__(WanT5EncoderLoader)
    loader._app_config = MagicMock(log_memory_usage=False, models_path=root)
    loader._torch_device = torch.device(loader_device or device)
    loader._ram_cache = cache
    loader._logger = logger
    return loader


def _retained_bytes(model: UMT5EncoderModel) -> int:
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in [*model.parameters(), *model.buffers()]
    }
    return sum(storages.values())


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("sharded", [False, True], ids=["single", "sharded"])
def test_cpu_load_widens_stored_weights_and_encodes(tmp_path: Path, nested: bool, sharded: bool) -> None:
    stored = _write_encoder(tmp_path / "text_encoder" if nested else tmp_path, sharded)
    loader = _loader(tmp_path)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    config = _config(Path("."))

    with loader.load_model(config, ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        assert model.encoder.embed_tokens.weight is model.shared.weight
        make_room.assert_called_once_with(_retained_bytes(model))
        expected = UMT5EncoderModel(stored.config).eval()
        expected.load_state_dict({key: value.float() for key, value in stored.state_dict().items()})
        with torch.no_grad():
            ids = torch.tensor([[3, 7, 1, 30, 12, 0]])
            torch.testing.assert_close(
                model(input_ids=ids).last_hidden_state, expected(input_ids=ids).last_hidden_state
            )


@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu_first", "accelerator_first"])
def test_shared_cache_does_not_adopt_weights_of_another_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_first: bool
) -> None:
    _write_encoder(tmp_path)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.bfloat16))
    store = SharedCpuWeightsStore()
    budget = RamBudget(max_bytes=10**12, shared_store=store)
    loaders = {
        "cpu": _loader(tmp_path, budget=budget, store=store),
        "accelerator": _loader(tmp_path, "cuda:0", budget, store),
    }
    order = ("cpu", "accelerator") if cpu_first else ("accelerator", "cpu")
    results = {}
    for role in order:
        config = _config(Path("."), cpu_only=role == "cpu")
        with loaders[role].load_model(config, ENCODER) as model:
            results[role] = {p.dtype for p in model.parameters()}
    assert results == {"cpu": {torch.float32}, "accelerator": {torch.bfloat16}}


@pytest.mark.parametrize(
    ("cache_device", "loader_device", "cpu_only", "dtype"),
    [
        ("cpu", "cuda:0", False, torch.float32),
        ("cuda:0", "cpu", False, torch.bfloat16),
        ("cuda:0", "cuda:0", False, torch.bfloat16),
        ("cuda:0", "cuda:0", True, torch.float32),
    ],
)
def test_precision_follows_the_model_override_or_selected_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_device: str,
    loader_device: str,
    cpu_only: bool,
    dtype: torch.dtype,
) -> None:
    _write_encoder(tmp_path)
    choose = MagicMock(return_value=torch.float16)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    loader = _loader(tmp_path, cache_device, loader_device=loader_device)
    with loader.load_model(_config(Path("."), cpu_only), ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {dtype}
    choose.assert_not_called()


@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prefixed", [False, True], ids=["plain_keys", "prefixed_keys"])
def test_admission_counts_distinct_embeddings_and_missing_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_dtype: torch.dtype, prefixed: bool
) -> None:
    _write_encoder(tmp_path, feed_forward_proj="relu")
    weights = tmp_path / "model.safetensors"
    state = load_file(weights)
    state["encoder.embed_tokens.weight"] = state["shared.weight"] + 1
    del state["encoder.block.0.layer.0.SelfAttention.q.weight"]
    if prefixed:
        state = {"transformer." + key: tensor for key, tensor in state.items()}
    save_file(state, weights, metadata={"format": "pt"})
    device = "cpu" if target_dtype == torch.float32 else "cuda:0"
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.bfloat16))
    loader = _loader(tmp_path, device)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)

    with loader.load_model(_config(Path(".")), ENCODER) as model:
        assert model.encoder.embed_tokens.weight is not model.shared.weight
        prefix = "transformer." if prefixed else ""
        torch.testing.assert_close(
            model.encoder.embed_tokens.weight, state[prefix + "encoder.embed_tokens.weight"].to(target_dtype)
        )
        torch.testing.assert_close(model.shared.weight, state[prefix + "shared.weight"].to(target_dtype))
        assert model.encoder.block[0].layer[0].SelfAttention.q.weight.device.type == "cpu"
        make_room.assert_called_once_with(_retained_bytes(model))


@pytest.mark.parametrize("selection", ["single_before_index", "explicit_single", "explicit_index"])
def test_admission_sizes_transformers_selected_weights(tmp_path: Path, selection: str) -> None:
    sharded = selection == "explicit_index"
    overrides = {"num_layers": 0, "custom_note": "x" * (1024 * 1024)} if selection == "explicit_single" else {}
    stored = _write_encoder(tmp_path, sharded, **overrides)
    default_index = tmp_path / "model.safetensors.index.json"
    default_weights = tmp_path / "model.safetensors"
    if selection == "single_before_index":
        default_index.write_text('{"metadata":{"total_size":1},"weight_map":{"stale":"missing.safetensors"}}')
    else:
        if sharded:
            index = json.loads(default_index.read_text(encoding="utf-8"))
            subdir = tmp_path / "weights"
            subdir.mkdir()
            for name in set(index["weight_map"].values()):
                (tmp_path / name).rename(subdir / name)
            index["weight_map"] = {key: f"weights/{name}" for key, name in index["weight_map"].items()}
            default_index.unlink()
            filename = "encoder.safetensors.index.json"
            (tmp_path / filename).write_text(json.dumps(index), encoding="utf-8")
        else:
            filename = "encoder.safetensors"
            default_weights.rename(tmp_path / filename)
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["transformers_weights"] = filename
        config_path.write_text(json.dumps(config), encoding="utf-8")
        save_file({"unused": torch.zeros(1)}, default_weights)
    torch.save({"unused": torch.zeros(10_000)}, tmp_path / "pytorch_model.bin")

    loader = _loader(tmp_path)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        for key, tensor in stored.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], tensor.float(), rtol=0, atol=0)
        make_room.assert_called_once_with(_retained_bytes(model))


@pytest.mark.parametrize("defect", ["config", "file", "shard", "explicit_file", "binary_shard", "header"])
def test_invalid_local_checkpoint_fails_before_making_room(tmp_path: Path, defect: str) -> None:
    _write_encoder(tmp_path, sharded=defect in ("shard", "binary_shard"))
    config_path = tmp_path / "config.json"
    weights = tmp_path / "model.safetensors"
    if defect == "config":
        config_path.unlink()
    elif defect == "file":
        weights.rename(tmp_path / "unknown.weights")
    elif defect == "explicit_file":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["transformers_weights"] = "missing.safetensors"
        config_path.write_text(json.dumps(config), encoding="utf-8")
    elif defect == "header":
        weights.write_bytes(b"corrupt")
    else:
        index_path = tmp_path / "model.safetensors.index.json"
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        if defect == "binary_shard":
            index_path.unlink()
            index_data["weight_map"] = dict.fromkeys(index_data["weight_map"], "missing.bin")
            (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps(index_data), encoding="utf-8")
        else:
            (tmp_path / next(iter(index_data["weight_map"].values()))).unlink()

    loader = _loader(tmp_path)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with pytest.raises((OSError, ValueError, RuntimeError, SafetensorError)):
        loader.load_model(_config(Path(".")), ENCODER)
    make_room.assert_not_called()


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors.index.json"])
def test_fresh_loader_reuses_a_warm_encoder_when_metadata_is_unavailable(tmp_path: Path, missing: str) -> None:
    _write_encoder(tmp_path, sharded=True)
    loader = _loader(tmp_path)
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        first = model

    (tmp_path / missing).unlink()
    fresh = _loader(tmp_path)
    fresh._ram_cache = loader.ram_cache
    fresh.get_size_fs = MagicMock(side_effect=AssertionError("cache hit ran the cold size check"))
    with fresh.load_model(_config(Path(".")), ENCODER) as model:
        assert model is first


@pytest.mark.parametrize("compute_device", ["cpu", "cuda:0"])
def test_warm_legacy_entry_when_metadata_is_unavailable_respects_cpu_only(tmp_path: Path, compute_device: str) -> None:
    model = _write_encoder(tmp_path)
    cache = _loader(tmp_path, "cuda:0").ram_cache
    key = get_model_cache_key(_config(tmp_path).key, ENCODER)
    cache.put(key, model, execution_device=torch.device(compute_device))
    record = cache.get(key)
    assert record.cached_model.compute_device == torch.device(compute_device)

    config_path = tmp_path / "config.json"
    config_path.unlink()
    fresh = _loader(tmp_path, "cuda:0")
    fresh._ram_cache = cache
    if compute_device == "cpu":
        with fresh.load_model(_config(Path("."), cpu_only=True), ENCODER) as loaded:
            assert loaded is model
            assert {p.dtype for p in loaded.parameters()} == {torch.bfloat16}
    else:
        with pytest.raises(FileNotFoundError, match="Wan UMT5 configuration not found") as rejected:
            fresh.load_model(_config(Path("."), cpu_only=True), ENCODER)
        assert str(config_path) in str(rejected.value)
        # The traceback still owns the loader frame, so dropping the claim later is insufficient.
        assert record.first_use_holds == 0


@pytest.mark.parametrize("key", ["encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"])
def test_wrong_known_weight_shape_fails_before_making_room(tmp_path: Path, key: str) -> None:
    _write_encoder(tmp_path)
    path = tmp_path / "model.safetensors"
    state = {name: tensor.clone() for name, tensor in load_file(path).items()}
    state[key] = state[key][1:].contiguous()
    save_file(state, path, metadata={"format": "pt"})
    loader = _loader(tmp_path)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)

    with pytest.raises((ValueError, RuntimeError)):
        loader.load_model(_config(Path(".")), ENCODER)
    make_room.assert_not_called()


def test_unknown_checkpoint_keys_keep_transformers_behavior(tmp_path: Path) -> None:
    stored = _write_encoder(tmp_path)
    path = tmp_path / "model.safetensors"
    state = {name: tensor.clone() for name, tensor in load_file(path).items()}
    state["encoder.block.99.layer.0.SelfAttention.q.weight"] = torch.zeros(1)
    save_file(state, path, metadata={"format": "pt"})
    with _loader(tmp_path).load_model(_config(Path(".")), ENCODER) as model:
        torch.testing.assert_close(model.shared.weight, stored.shared.weight.float(), rtol=0, atol=0)


@pytest.mark.parametrize("index_name", ["model.safetensors.index.json", "pytorch_model.bin.index.json"])
def test_shard_dispatch_follows_the_first_resolved_file(tmp_path: Path, index_name: str) -> None:
    stored = _write_encoder(tmp_path, sharded=True)
    index = (tmp_path / "model.safetensors.index.json").rename(tmp_path / index_name)
    data = json.loads(index.read_text(encoding="utf-8"))
    shards = sorted(set(data["weight_map"].values()))
    assert len(shards) > 1
    last = tmp_path / shards[-1]
    uppercase = last.with_suffix(".SAFETENSORS")
    last.rename(last.with_suffix(".tmp")).rename(uppercase)
    data["weight_map"] = {
        key: uppercase.name if name == last.name else name for key, name in data["weight_map"].items()
    }
    index.write_text(json.dumps(data), encoding="utf-8")

    loader = _loader(tmp_path)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        torch.testing.assert_close(model.shared.weight, stored.shared.weight.float(), rtol=0, atol=0)
        make_room.assert_called_once_with(_retained_bytes(model))


def test_unavailable_metadata_cannot_cold_load_under_the_legacy_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_encoder(tmp_path)
    loader = _loader(tmp_path)
    read = loader._read_checkpoint

    def metadata_returns_after_lookup(folder: Path):
        monkeypatch.setattr(loader, "_read_checkpoint", read)
        raise FileNotFoundError("config temporarily unavailable")

    monkeypatch.setattr(loader, "_read_checkpoint", metadata_returns_after_lookup)
    put = loader.ram_cache.put = MagicMock(wraps=loader.ram_cache.put)
    with pytest.raises(FileNotFoundError, match="config temporarily unavailable"):
        loader.load_model(_config(Path(".")), ENCODER)
    put.assert_not_called()
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_nested_path_disappearing_after_key_selection_does_not_load_a_root_binary(tmp_path: Path) -> None:
    nested = tmp_path / "text_encoder"
    stored = _write_encoder(nested)
    stored.config.save_pretrained(tmp_path)
    torch.save(stored.state_dict(), tmp_path / "pytorch_model.bin")
    loader = _loader(tmp_path)
    get_size = loader.get_size_fs

    def remove_nested_before_admission(*args):
        nested.rename(tmp_path / "temporarily-away")
        return get_size(*args)

    loader.get_size_fs = remove_nested_before_admission
    put = loader.ram_cache.put = MagicMock(wraps=loader.ram_cache.put)
    with pytest.raises(FileNotFoundError):
        loader.load_model(_config(Path(".")), ENCODER)
    put.assert_not_called()


@pytest.mark.parametrize("sharded", [False, True], ids=["single", "sharded"])
def test_binary_folder_keeps_inherited_bfloat16_cache_and_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sharded: bool
) -> None:
    model = _write_encoder(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    state = model.state_dict()
    if sharded:
        keys = list(state)
        weight_map = {}
        for shard, group in enumerate((keys[::2], keys[1::2])):
            filename = f"pytorch_model-{shard}.bin"
            torch.save({key: state[key] for key in group}, tmp_path / filename)
            weight_map.update(dict.fromkeys(group, filename))
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps({"metadata": {"total_size": sum(t.nbytes for t in state.values())}, "weight_map": weight_map}),
            encoding="utf-8",
        )
    else:
        key = "encoder.block.0.layer.0.SelfAttention.q.weight"
        original = state[key]
        padded = torch.empty(32_768, dtype=original.dtype)
        state[key] = padded[: original.numel()].view_as(original).copy_(original)
        torch.save(state, tmp_path / "pytorch_model.bin")
    choose = MagicMock(side_effect=AssertionError("legacy path should skip the dtype policy"))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    loader = _loader(tmp_path, "cuda:0")
    config = _config(tmp_path, cpu_only=True)
    make_room = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)

    assert loader._cache_key_variant(config, ENCODER) == ""
    with loader.load_model(config, ENCODER) as loaded:
        assert {p.dtype for p in loaded.parameters()} == {torch.bfloat16}
        torch.testing.assert_close(loaded.shared.weight, state["shared.weight"], rtol=0, atol=0)
    make_room.assert_called_once_with(ModelLoader.get_size_fs(loader, config, tmp_path, ENCODER))
    choose.assert_not_called()
