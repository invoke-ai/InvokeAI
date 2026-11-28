"""CPU and simulated accelerator coverage for standalone Wan UMT5 folder loading."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import UMT5Config, UMT5EncoderModel

from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_WanT5Encoder_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
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
    choose = MagicMock(return_value=torch.bfloat16)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    loader = _loader(tmp_path, cache_device, loader_device=loader_device)
    with loader.load_model(_config(Path("."), cpu_only), ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {dtype}


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


@pytest.mark.parametrize("key", ["encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"])
def test_wrong_known_weight_shape_is_rejected_by_transformers(tmp_path: Path, key: str) -> None:
    _write_encoder(tmp_path)
    path = tmp_path / "model.safetensors"
    state = {name: tensor.clone() for name, tensor in load_file(path).items()}
    state[key] = state[key][1:].contiguous()
    save_file(state, path, metadata={"format": "pt"})
    loader = _loader(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        loader.load_model(_config(Path(".")), ENCODER)


def test_unknown_checkpoint_keys_keep_transformers_behavior(tmp_path: Path) -> None:
    stored = _write_encoder(tmp_path)
    path = tmp_path / "model.safetensors"
    state = {name: tensor.clone() for name, tensor in load_file(path).items()}
    state["encoder.block.99.layer.0.SelfAttention.q.weight"] = torch.zeros(1)
    save_file(state, path, metadata={"format": "pt"})
    with _loader(tmp_path).load_model(_config(Path(".")), ENCODER) as model:
        torch.testing.assert_close(model.shared.weight, stored.shared.weight.float(), rtol=0, atol=0)


def test_concurrent_cold_loads_share_one_device_variant(tmp_path: Path) -> None:
    _write_encoder(tmp_path)
    loaders = [_loader(tmp_path), _loader(tmp_path)]
    loaders[1]._ram_cache = loaders[0].ram_cache
    put = loaders[0].ram_cache.put = MagicMock(wraps=loaders[0].ram_cache.put)
    barrier = Barrier(2)
    for loader in loaders:
        original = loader._cache_key_variant

        def synchronized_key(config, submodel_type, original=original):
            variant = original(config, submodel_type)
            barrier.wait(timeout=15)
            return variant

        loader._cache_key_variant = synchronized_key

    def load(loader):
        with loader.load_model(_config(Path(".")), ENCODER) as model:
            return model

    with ThreadPoolExecutor(max_workers=2) as pool:
        models = list(pool.map(load, loaders))
    assert models[0] is models[1]
    put.assert_called_once()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_binary_estimate_uses_file_size_and_device_dtype(
    tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _write_encoder(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.bfloat16))
    weights = tmp_path / "pytorch_model.bin"
    torch.save(stored.state_dict(), weights)
    loader = _loader(tmp_path, device)
    reserve = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        expected_dtype = torch.float32 if device == "cpu" else torch.bfloat16
        assert {p.dtype for p in model.parameters()} == {expected_dtype}
        torch.testing.assert_close(model.shared.weight, stored.shared.weight.to(expected_dtype), rtol=0, atol=0)
        reserve.assert_called_once_with(weights.stat().st_size * (2 if device == "cpu" else 1))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_safetensors_estimate_uses_target_dtype_not_storage_dtype(
    tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_encoder(tmp_path)
    weights = tmp_path / "model.safetensors"
    state = {key: tensor.float() for key, tensor in load_file(weights).items()}
    save_file(state, weights, metadata={"format": "pt"})
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.bfloat16))
    loader = _loader(tmp_path, device)
    reserve = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        assert {p.dtype for p in model.parameters()} == {dtype}
        reserve.assert_called_once_with(sum(t.numel() for t in state.values()) * dtype.itemsize)


def test_accelerator_without_bfloat16_uses_float32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_encoder(tmp_path)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    loader = _loader(tmp_path, "mps")
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.parametrize("cpu_first", [True, False])
def test_cpu_override_does_not_reuse_float32_accelerator_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_first: bool
) -> None:
    _write_encoder(tmp_path)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    loader = _loader(tmp_path, "mps")
    configs = [_config(Path("."), cpu_only=cpu_first), _config(Path("."), cpu_only=not cpu_first)]
    models = []
    for config in configs:
        with loader.load_model(config, ENCODER) as model:
            models.append(model)
            assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert models[0] is not models[1]


def test_dtype_decision_is_stable_through_cache_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_encoder(tmp_path)
    choose = MagicMock(side_effect=[torch.float16, torch.bfloat16])
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(choose))
    loader = _loader(tmp_path, "mps")
    config = _config(Path("."))
    key = loader._cache_key_variant(config, ENCODER)
    with loader.load_model(config, ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert loader._cache_key_variant(config, ENCODER) == key
    choose.assert_called_once_with(torch.device("mps"))


@pytest.mark.parametrize("setting", ["quantization_config", "fusion_config"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_quantized_or_fused_folders_keep_existing_dtype_and_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str, binary: bool, device: str
) -> None:
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.wan.read_safetensors_header",
        MagicMock(side_effect=AssertionError("quantized and fused headers must not be sized as dense")),
    )
    model = _write_encoder(tmp_path)
    if binary:
        (tmp_path / "model.safetensors").unlink()
        torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    config_file = tmp_path / "config.json"
    data = json.loads(config_file.read_text(encoding="utf-8"))
    data[setting] = {}
    config_file.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    from_pretrained = MagicMock(return_value=model)
    monkeypatch.setattr(UMT5EncoderModel, "from_pretrained", from_pretrained)
    loader = _loader(tmp_path, device)
    config = _config(tmp_path)
    assert loader.get_size_fs(config, tmp_path, ENCODER) == ModelLoader.get_size_fs(loader, config, tmp_path, ENCODER)
    assert loader._load_model(config, ENCODER) is model
    assert from_pretrained.call_args.kwargs["dtype"] == torch.bfloat16


def test_binary_accelerator_float32_fallback_keeps_gpu_file_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _write_encoder(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    weights = tmp_path / "pytorch_model.bin"
    torch.save(stored.state_dict(), weights)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float16))
    loader = _loader(tmp_path, "mps")
    reserve = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with loader.load_model(_config(Path(".")), ENCODER) as model:
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        reserve.assert_called_once_with(weights.stat().st_size)


def test_adapter_only_keeps_main_loading_and_sizing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    model = MagicMock()
    from_pretrained = MagicMock(return_value=model)
    monkeypatch.setattr(UMT5EncoderModel, "from_pretrained", from_pretrained)
    loader = _loader(tmp_path)
    config = _config(tmp_path)
    assert loader.get_size_fs(config, tmp_path, ENCODER) == ModelLoader.get_size_fs(loader, config, tmp_path, ENCODER)
    assert loader._load_model(config, ENCODER) is model
    assert from_pretrained.call_args.kwargs["dtype"] == torch.bfloat16


def test_real_8bit_cpu_folder_keeps_main_loading_and_sizing(tmp_path: Path) -> None:
    pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig

    base = tmp_path / "base"
    quantized = tmp_path / "quantized"
    _write_encoder(base)
    original = UMT5EncoderModel.from_pretrained(
        str(base), quantization_config=BitsAndBytesConfig(load_in_8bit=True), dtype=torch.float32, device_map="cpu"
    )
    original.save_pretrained(quantized)
    loader = _loader(tmp_path)
    config = _config(quantized)
    assert loader.get_size_fs(config, quantized, ENCODER) == ModelLoader.get_size_fs(loader, config, quantized, ENCODER)
    reference = UMT5EncoderModel.from_pretrained(str(quantized), dtype=torch.bfloat16, local_files_only=True)
    with loader.load_model(config, ENCODER) as model:
        assert [(name, p.dtype) for name, p in model.named_parameters()] == [
            (name, p.dtype) for name, p in reference.named_parameters()
        ]
        with torch.no_grad():
            ids = torch.tensor([[3, 7, 1]])
            torch.testing.assert_close(
                model(input_ids=ids).last_hidden_state, reference(input_ids=ids).last_hidden_state
            )


@pytest.mark.parametrize(
    "kind", ["safe-index-binary", "binary-index-safe", "binary-first-mixed", "safe-first-uppercase"]
)
def test_transformers_shard_dispatch_and_estimate(tmp_path: Path, kind: str) -> None:
    stored = _write_encoder(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    state = {k: v.contiguous().clone() for k, v in stored.state_dict().items()}
    keys = list(state)
    if kind == "safe-index-binary":
        index = "model.safetensors.index.json"
        shards = [("a.bin", keys)]
    elif kind == "binary-index-safe":
        index = "pytorch_model.bin.index.json"
        shards = [("a.safetensors", keys)]
    elif kind == "binary-first-mixed":
        index = "model.safetensors.index.json"
        shards = [("a.bin", keys[::2]), ("b.safetensors", keys[1::2])]
    else:
        index = "pytorch_model.bin.index.json"
        shards = [("a.safetensors", keys[::2]), ("b.SAFETENSORS", keys[1::2])]
    expected = 0
    weight_map = {}
    for name, group in shards:
        path = tmp_path / name
        values = {k: state[k] for k in group}
        if name.lower().endswith(".safetensors"):
            save_file(values, path, metadata={"format": "pt"})
            expected += sum(t.numel() for t in values.values()) * torch.float32.itemsize
        else:
            torch.save(values, path)
            expected += path.stat().st_size * 2
        weight_map.update(dict.fromkeys(group, name))
    (tmp_path / index).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    reference = UMT5EncoderModel.from_pretrained(str(tmp_path), dtype=torch.float32, local_files_only=True)
    loader = _loader(tmp_path)
    reserve = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    with loader.load_model(_config(tmp_path), ENCODER) as actual:
        torch.testing.assert_close(actual.shared.weight, reference.shared.weight)
        reserve.assert_called_once_with(expected)


@pytest.mark.parametrize("embedded", [False, True], ids=["redirect", "embedded"])
def test_peft_folder_keeps_transformers_main_path(tmp_path: Path, embedded: bool) -> None:
    peft = pytest.importorskip("peft")
    base_dir = tmp_path / "base"
    adapter_dir = base_dir if embedded else tmp_path / "adapter"
    base = _write_encoder(base_dir, vocab_size=512)
    adapter = peft.get_peft_model(base, peft.LoraConfig(r=16, lora_alpha=16, target_modules=["q"]))
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.05)
    adapter.save_pretrained(adapter_dir)
    adapter_config = adapter_dir / "adapter_config.json"
    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    # Transformers must use the local base when the model config is present.
    data["base_model_name_or_path"] = str(tmp_path / "missing-base" if embedded else base_dir)
    adapter_config.write_text(json.dumps(data), encoding="utf-8")
    reference = UMT5EncoderModel.from_pretrained(str(adapter_dir), dtype=torch.bfloat16, local_files_only=True)
    loader = _loader(tmp_path)
    config = _config(adapter_dir)
    reserve = loader.ram_cache.make_room = MagicMock(wraps=loader.ram_cache.make_room)
    estimate = ModelLoader.get_size_fs(loader, config, adapter_dir, ENCODER)
    with loader.load_model(config, ENCODER) as actual:
        assert any("lora_" in name for name, _ in actual.named_parameters())
        assert [(name, p.dtype) for name, p in actual.named_parameters()] == [
            (name, p.dtype) for name, p in reference.named_parameters()
        ]
        with torch.no_grad():
            ids = torch.tensor([[3, 7, 1]])
            torch.testing.assert_close(
                actual(input_ids=ids).last_hidden_state, reference(input_ids=ids).last_hidden_state
            )
        reserve.assert_called_once_with(estimate)


def test_adapter_metadata_is_ignored_when_peft_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_encoder(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("transformers.utils.is_peft_available", lambda: False)
    from_pretrained = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(UMT5EncoderModel, "from_pretrained", from_pretrained)
    loader = _loader(tmp_path)
    config = _config(tmp_path)
    assert loader.get_size_fs(config, tmp_path, ENCODER) != ModelLoader.get_size_fs(loader, config, tmp_path, ENCODER)
    loader._load_model(config, ENCODER)
    assert from_pretrained.call_args.kwargs["dtype"] == torch.float32
