"""Tests for load-time adoption of shared CPU weights (multi-GPU RAM-spike fix).

When a second device loads a model that another device already holds, the loader deep-copies the
empty (meta-weight) structural shell the first device registered and assigns the canonical CPU
weights into it — instead of re-reading the model from disk and materializing a full transient
second copy. This is loader-agnostic (no per-model-family code): it works by cloning a built module,
so it covers diffusers, single-file checkpoints, GGUF and transformers models alike, and preserves
any registered hooks (e.g. fp8 layerwise-cast hooks).
"""

from unittest.mock import MagicMock

import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        # A non-persistent buffer: not in the state dict, so adoption must carry it over with data.
        self.register_buffer("scale", torch.tensor([2.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * self.scale


def _loader_with_store(store: SharedCpuWeightsStore | None) -> ModelLoader:
    loader = ModelLoader.__new__(ModelLoader)  # bypass __init__ (needs app deps we don't use here)
    loader._logger = MagicMock()
    loader._ram_cache = MagicMock()
    loader._ram_cache.shared_cpu_weights = store
    return loader


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


def test_build_meta_shell_returns_none_for_non_module():
    assert ModelLoader._build_meta_shell({"not": "a module"}) is None  # type: ignore[arg-type]


def test_adopts_canonical_weights_without_copying():
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    _populate(store, "m", source)
    canonical = store.peek("m")
    refcount_before = store.refcount("m")

    model = _loader_with_store(store)._try_adopt_shared_weights("m")

    assert model is not None
    # The adopted params ARE the canonical tensors (assign=True, no copy) -> no extra RAM.
    assert model.lin.weight.data_ptr() == canonical["lin.weight"].data_ptr()
    assert model.lin.bias.data_ptr() == canonical["lin.bias"].data_ptr()
    assert not any(t.is_meta for t in model.parameters())
    assert not any(t.is_meta for t in model.buffers())
    # peek()/get_shell() must not have taken a reference -- the wrapper's acquire() does that later.
    assert store.refcount("m") == refcount_before


def test_adopted_model_produces_correct_output():
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    _populate(store, "m", source)
    x = torch.randn(3, 4)

    model = _loader_with_store(store)._try_adopt_shared_weights("m")

    assert torch.allclose(model(x), source(x), atol=1e-6)


def test_adoption_preserves_forward_hooks():
    # fp8 layerwise casting is implemented as forward hooks; cloning the built module must keep them.
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    fired: list[str] = []
    source.lin.register_forward_pre_hook(lambda mod, args: fired.append("pre"))
    _populate(store, "m", source)

    model = _loader_with_store(store)._try_adopt_shared_weights("m")
    model(torch.randn(1, 4))

    assert fired == ["pre"]  # the cloned module's hook fired


def test_no_shell_means_no_adoption():
    # Canonical present but no shell registered (e.g. first device couldn't clone) -> fall back.
    store = SharedCpuWeightsStore()
    store.acquire("m", _TinyModel().state_dict())
    assert _loader_with_store(store)._try_adopt_shared_weights("m") is None


def test_absent_key_means_no_adoption():
    assert _loader_with_store(SharedCpuWeightsStore())._try_adopt_shared_weights("missing") is None


def test_no_shared_store_means_no_adoption():
    assert _loader_with_store(None)._try_adopt_shared_weights("m") is None


def test_mismatched_canonical_falls_back_safely():
    # If the canonical weights don't match the shell's structure, adoption must fail soft (-> None),
    # not raise, so the caller can load normally.
    store = SharedCpuWeightsStore()
    source = _TinyModel()
    shell = ModelLoader._build_meta_shell(source)
    assert shell is not None
    store.acquire("m", {"unexpected.key": torch.zeros(2)})  # wrong state dict
    store.set_shell("m", shell)

    loader = _loader_with_store(store)
    assert loader._try_adopt_shared_weights("m") is None
    loader._logger.warning.assert_called_once()


def test_shell_dropped_when_entry_released():
    store = SharedCpuWeightsStore()
    _populate(store, "m", _TinyModel())
    assert store.get_shell("m") is not None
    store.release("m")  # last reference -> entry (and its shell) gone
    assert store.get_shell("m") is None
    assert "m" not in store
