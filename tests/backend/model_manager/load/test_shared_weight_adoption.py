"""Tests for load-time adoption of shared CPU weights (multi-GPU RAM-spike fix).

When a second device loads a model that another device already holds, the loader deep-copies the
empty (meta-weight) structural shell the first device registered and assigns the canonical CPU
weights into it — instead of re-reading the model from disk and materializing a full transient
second copy. This is loader-agnostic (no per-model-family code): it works by cloning a built module,
so it covers diffusers, single-file checkpoints, GGUF and transformers models alike, and preserves
any registered hooks (e.g. fp8 layerwise-cast hooks).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache import model_cache as cache_module
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.ram_budget import RamBudget
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        # A non-persistent buffer: not in the state dict, so adoption must carry it over with data.
        self.register_buffer("scale", torch.tensor([2.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * self.scale


def _adopt(store: SharedCpuWeightsStore, key: str):
    loader = ModelLoader.__new__(ModelLoader)  # bypass __init__ (needs app deps we don't use here)
    loader._logger = MagicMock()
    with store.pin_for_adoption(key) as pin:
        return loader._try_adopt_shared_weights(key, pin) if pin is not None else None


def _populate(store: SharedCpuWeightsStore, key: str, model: torch.nn.Module) -> None:
    """Mimic the first device's load: register canonical weights + a meta shell for `model`."""
    store.acquire(key, model.state_dict())
    shell = ModelLoader._build_meta_shell(model)
    assert shell is not None
    store.set_shell(key, shell)


def test_meta_shell_has_no_real_weight_storage():
    model = _TinyModel()
    shell = ModelLoader._build_meta_shell(model)
    assert shell is not None
    # Parameters are on meta (0 bytes); the non-persistent buffer keeps real data.
    assert all(p.is_meta for p in shell.parameters())
    assert not shell.scale.is_meta
    assert torch.equal(shell.scale, model.scale)


def test_adopts_gguf_quantized_weights_and_shares_storage():
    """Regression: GGUF models (GGMLTensor params) must be adoptable.

    `GGMLTensor.__torch_dispatch__` returns NotImplemented for `aten.empty_like`, so the original
    `_build_meta_shell` threw on the first parameter and silently returned None -> no shell -> the
    second GPU re-loaded the (largest) quantized model from disk on every run, spiking RAM. The
    meta-shell builder must fall back to a plain meta placeholder, and adoption must share the
    quantized storage (the whole point of dedup).
    """
    import math

    import gguf

    from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

    def _make_q8(logical_shape: tuple[int, int]) -> GGMLTensor:
        n = math.prod(logical_shape)
        nbytes = (n // 32) * 34  # Q8_0: 34 bytes per 32-element block
        return GGMLTensor(
            torch.zeros(nbytes, dtype=torch.uint8),
            gguf.GGMLQuantizationType.Q8_0,
            torch.Size(logical_shape),
            torch.bfloat16,
        )

    model = _TinyModel()
    # Swap the real weight for a GGMLTensor, mirroring the gguf loader's load_state_dict(assign=True).
    model.lin.weight = torch.nn.Parameter(_make_q8((4, 4)), requires_grad=False)
    orig_storage = model.lin.weight.quantized_data.untyped_storage().data_ptr()

    # The shell must build despite GGMLTensor not supporting empty_like.
    shell = ModelLoader._build_meta_shell(model)
    assert shell is not None
    assert shell.lin.weight.is_meta

    store = SharedCpuWeightsStore()
    store.acquire("g", model.state_dict())
    store.set_shell("g", shell)

    adopted = _adopt(store, "g")

    assert adopted is not None
    assert not any(t.is_meta for t in adopted.parameters())
    # The adopted GGMLTensor must share the quantized blob's storage -> one RAM copy across devices.
    assert isinstance(adopted.lin.weight, GGMLTensor)
    assert adopted.lin.weight.quantized_data.untyped_storage().data_ptr() == orig_storage


def test_build_meta_shell_returns_none_for_non_module():
    assert ModelLoader._build_meta_shell({"not": "a module"}) is None  # type: ignore[arg-type]


def test_adopts_canonical_weights_without_copying():
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    _populate(store, "m", source)
    canonical = store.peek("m")
    refcount_before = store.refcount("m")

    model = _adopt(store, "m")

    assert model is not None
    # The adopted params ARE the canonical tensors (assign=True, no copy) -> no extra RAM.
    assert model.lin.weight.data_ptr() == canonical["lin.weight"].data_ptr()
    assert model.lin.bias.data_ptr() == canonical["lin.bias"].data_ptr()
    assert not any(t.is_meta for t in model.parameters())
    assert not any(t.is_meta for t in model.buffers())
    # The temporary pin is gone after leaving its scope.
    assert store.refcount("m") == refcount_before


def test_adopted_model_produces_correct_output():
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    _populate(store, "m", source)
    x = torch.randn(3, 4)

    model = _adopt(store, "m")

    assert torch.allclose(model(x), source(x), atol=1e-6)


def test_adoption_preserves_forward_hooks():
    # fp8 layerwise casting is implemented as forward hooks; cloning the built module must keep them.
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    fired: list[str] = []
    source.lin.register_forward_pre_hook(lambda mod, args: fired.append("pre"))
    _populate(store, "m", source)

    model = _adopt(store, "m")
    model(torch.randn(1, 4))

    assert fired == ["pre"]  # the cloned module's hook fired


def test_no_shell_means_no_adoption():
    # Canonical present but no shell registered (e.g. first device couldn't clone) -> fall back.
    store = SharedCpuWeightsStore()
    store.acquire("m", _TinyModel().state_dict())
    assert _adopt(store, "m") is None


def test_absent_key_means_no_adoption():
    assert _adopt(SharedCpuWeightsStore(), "missing") is None


def test_mismatched_canonical_falls_back_safely():
    # If the canonical weights don't match the shell's structure, adoption must fail soft (-> None),
    # not raise, so the caller can load normally.
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    shell = ModelLoader._build_meta_shell(source)
    assert shell is not None
    store.acquire("m", {"unexpected.key": torch.zeros(2)})  # wrong state dict
    store.set_shell("m", shell)

    assert _adopt(store, "m") is None
    assert store.refcount("m") == 1


def test_shell_dropped_when_entry_released():
    store = SharedCpuWeightsStore()
    _populate(store, "m", _TinyModel())
    assert store.get_shell("m") is not None
    store.release("m")  # last reference -> entry (and its shell) gone
    assert store.get_shell("m") is None
    assert "m" not in store


class _AdmissionModel(torch.nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)
        if layout == "private":
            self.register_buffer("private", torch.ones(8), persistent=False)
        if layout == "tied":
            self.head = torch.nn.Linear(8, 8, bias=False)
            self.head.weight = self.proj.weight


class _FileLoader(ModelLoader):
    def __init__(self, cache, path, layout):
        self._ram_cache = cache
        self._logger = logging.getLogger(__name__)
        self._app_config = SimpleNamespace(log_memory_usage=False, models_path=path.parent)
        self.layout = layout
        self.disk_loads = 0

    def _load_model(self, config, submodel_type=None):
        self.disk_loads += 1
        model = _AdmissionModel(self.layout)
        model.load_state_dict(torch.load(config.path, weights_only=True))
        return model


@pytest.fixture
def adoption_pair(tmp_path):
    caches = []

    def create(*, layout="dense", keep_copy=True, cap=10**12):
        path = tmp_path / "tiny.pt"
        torch.save(_AdmissionModel(layout).state_dict(), path)
        store = SharedCpuWeightsStore()
        budget = RamBudget(max_bytes=cap, shared_store=store)
        for keep in (True, keep_copy):
            caches.append(
                ModelCache(
                    execution_device_working_mem_gb=1,
                    enable_partial_loading=False,
                    keep_ram_copy_of_weights=keep,
                    execution_device="cpu",
                    storage_device="cpu",
                    logger=logging.getLogger(__name__),
                    shared_cpu_weights=store,
                    ram_budget=budget,
                )
            )
        source, dest = (_FileLoader(cache, path, layout) for cache in caches)
        config = SimpleNamespace(key="encoder", path=str(path), name="encoder", base="any", type="test")
        first, claim = source._load_and_cache(config)
        assert claim is not None
        claim.release()
        caches[0].lock(first, None)
        caches[0].unlock(first)
        return source, dest, config, store, budget

    yield create
    for cache in caches:
        cache.shutdown()


def test_adoption_keeps_peer_cached_and_avoids_reload(adoption_pair):
    source, dest, config, store, budget = adoption_pair(cap=356)
    first = source._ram_cache.get(config.key)
    canonical = store.peek(config.key)
    record, claim = dest._load_and_cache(config)
    assert claim is not None
    claim.release()
    dest._ram_cache.lock(record, None)
    dest._ram_cache.unlock(record)
    first_again, claim = source._load_and_cache(config)
    if claim is not None:
        claim.release()
    assert first_again is first
    assert source.disk_loads == 1 and dest.disk_loads == 0
    assert record.cached_model.get_cpu_state_dict() is canonical
    assert store.refcount(config.key) == 2
    assert budget.total_in_use() == 256


def test_failed_clone_releases_pin_and_loads_from_disk(adoption_pair, monkeypatch):
    source, dest, config, store, _ = adoption_pair()
    canonical = store.peek(config.key)
    assert canonical is not None

    def fail_clone(self, memo):
        raise RuntimeError("cannot copy shell")

    monkeypatch.setattr(_AdmissionModel, "__deepcopy__", fail_clone, raising=False)
    record, claim = dest._load_and_cache(config)
    assert claim is not None
    claim.release()
    assert source.disk_loads == 1 and dest.disk_loads == 1
    assert record.cached_model.get_cpu_state_dict() is canonical
    assert store.refcount(config.key) == 2


@pytest.mark.parametrize(
    ("layout", "policy", "reservation"),
    [("private", "keep", 288), ("tied", "keep", 512), ("dense", "no_copy", 256), ("dense", "integrated_xpu", 256)],
)
def test_full_reservation_can_evict_source_without_evicting_warm_peer(
    adoption_pair, monkeypatch, layout, policy, reservation
):
    warm_bytes = 256
    source, dest, config, store, budget = adoption_pair(
        layout=layout, keep_copy=policy != "no_copy", cap=reservation + warm_bytes
    )
    if policy == "integrated_xpu":
        # Exercise the integrated-device RAM-copy override without selecting an accelerator.
        dest_device = dest._ram_cache._execution_device
        monkeypatch.setattr(cache_module, "_is_integrated_xpu", lambda device: device is dest_device)
    warm_key = "warm"
    source._ram_cache.put(warm_key, torch.nn.Linear(8, 8, bias=False))
    warm = source._ram_cache.get(warm_key)
    source._ram_cache.lock(warm, None)
    source._ram_cache.unlock(warm)
    canonical = store.peek(config.key)
    assert canonical is not None
    assert store.get_shell(config.key) is not None

    record, claim = dest._load_and_cache(config)
    assert claim is not None
    claim.release()
    assert dest.disk_loads == 0
    assert source._ram_cache.get(warm_key) is warm
    with pytest.raises(IndexError):
        source._ram_cache.get(config.key)
    assert torch.equal(record.cached_model.model.proj.weight, canonical["proj.weight"])
    if policy == "keep":
        # Full reservation can evict the original entry. The adopted model is then registered
        # without a shell, as it was before the adoption pin was introduced.
        current = store.peek(config.key)
        assert current is not None and current is not canonical
        assert record.cached_model.get_cpu_state_dict() is current
    else:
        assert record.cached_model.get_cpu_state_dict() is None
        assert store.peek(config.key) is None
    assert store.get_shell(config.key) is None
    assert budget.total_in_use() <= budget.max_bytes


@pytest.mark.parametrize("fail_make_room", [False, True])
def test_matching_loader_pins_canonical_until_admission(adoption_pair, monkeypatch, fail_make_room):
    source, dest, config, store, budget = adoption_pair()
    key = config.key
    canonical, shell = store.peek(key), store.get_shell(key)
    assert canonical is not None and shell is not None

    def evict_last_peer():
        assert source._ram_cache.evict_unlocked_for_peer(lambda: False) == 1
        assert store.peek(key) is canonical
        assert store.get_shell(key) is shell

    if fail_make_room:

        def fail_reservation(size):
            assert size == 0
            evict_last_peer()
            raise RuntimeError("cache reservation failed")

        monkeypatch.setattr(dest._ram_cache, "_make_room_internal", fail_reservation)
        with pytest.raises(RuntimeError, match="cache reservation failed"):
            dest._load_and_cache(config)
        with pytest.raises(IndexError):
            dest._ram_cache.get(key)
        assert store.peek(key) is None
        assert budget.total_in_use() == 0
    else:
        original_put = dest._ram_cache.put

        def evict_before_put(*args, **kwargs):
            evict_last_peer()
            return original_put(*args, **kwargs)

        monkeypatch.setattr(dest._ram_cache, "put", evict_before_put)
        record, claim = dest._load_and_cache(config)
        assert claim is not None
        claim.release()
        assert dest.disk_loads == 0
        assert record.cached_model.get_cpu_state_dict() is canonical
        assert store.get_shell(key) is shell
        assert store.refcount(key) == 1
