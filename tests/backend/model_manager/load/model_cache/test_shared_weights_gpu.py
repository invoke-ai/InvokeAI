"""Real-GPU validation of cross-device CPU-weight sharing.

These require two CUDA (incl. ROCm/HIP) devices. They prove the properties the CPU-only unit tests
cannot: that a module re-pointed at shared canonical CPU weights (a) loads onto its GPU and produces
correct inference output, and (b) survives two GPUs loading/unloading from the *same* shared CPU
state dict concurrently without corrupting each other's results.
"""

import copy
import logging
import threading
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule
from tests.backend.quantization.gguf.test_ggml_tensor import quantize_tensor

requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices."
)

DEVICE_A = "cuda:0"
DEVICE_B = "cuda:1"


def _mock_logger() -> MagicMock:
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


def _make_cache(store: SharedCpuWeightsStore, device: str, partial: bool) -> ModelCache:
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=partial,
        keep_ram_copy_of_weights=True,
        execution_device=device,
        storage_device="cpu",
        logger=_mock_logger(),
        shared_cpu_weights=store,
    )


@requires_two_gpus
@pytest.mark.parametrize("partial", [False, True])
def test_shared_weights_produce_correct_output_on_both_gpus(partial: bool):
    """A model loaded on two GPUs from one shared CPU copy must compute correct results on both."""
    torch.manual_seed(0)
    model_a = DummyModule()
    # model_b starts with DIFFERENT weights; sharing must overwrite them with model_a's canonical
    # weights (both keys map to the same logical model).
    torch.manual_seed(1)
    model_b = DummyModule()

    x = torch.randn(4, 10)
    # Reference output from model_a's original weights, computed before any cache/device mutation.
    reference = copy.deepcopy(model_a)(x)

    store = SharedCpuWeightsStore()
    cache_a = _make_cache(store, DEVICE_A, partial)
    cache_b = _make_cache(store, DEVICE_B, partial)
    try:
        cache_a.put("m", model_a)
        cache_b.put("m", model_b)

        # Single shared CPU copy across both devices.
        assert store.refcount("m") == 2
        assert cache_a.get("m").cached_model.get_cpu_state_dict() is cache_b.get("m").cached_model.get_cpu_state_dict()

        rec_a = cache_a.get("m")
        rec_b = cache_b.get("m")
        cache_a.lock(rec_a, None)
        cache_b.lock(rec_b, None)
        try:
            out_a = rec_a.cached_model.model(x.to(DEVICE_A))
            out_b = rec_b.cached_model.model(x.to(DEVICE_B))
        finally:
            cache_a.unlock(rec_a)
            cache_b.unlock(rec_b)

        # Both devices reproduce model_a's output (so model_b really adopted the shared weights).
        assert torch.allclose(out_a.cpu(), reference, atol=1e-5)
        assert torch.allclose(out_b.cpu(), reference, atol=1e-5)
    finally:
        cache_a.shutdown()
        cache_b.shutdown()


@requires_two_gpus
@pytest.mark.parametrize("wrapper_cls", [CachedModelOnlyFullLoad, CachedModelWithPartialLoad])
def test_concurrent_load_unload_from_shared_state_dict(wrapper_cls):
    """Two GPUs repeatedly loading/unloading from one shared CPU state dict must not corrupt each
    other. Each thread drives its own device's wrapper; the canonical CPU tensors are read-only and
    must stay intact across concurrent .to(device) reads and load_state_dict(assign=True) restores.
    """
    torch.manual_seed(0)
    model_a = DummyModule()
    torch.manual_seed(1)
    model_b = DummyModule()

    x = torch.randn(4, 10)
    reference = copy.deepcopy(model_a)(x)

    store = SharedCpuWeightsStore()

    def build(model, device):
        if wrapper_cls is CachedModelWithPartialLoad:
            return CachedModelWithPartialLoad(
                model, torch.device(device), keep_ram_copy=True, shared_store=store, cache_key="m"
            )
        return CachedModelOnlyFullLoad(
            model, torch.device(device), total_bytes=1000, keep_ram_copy=True, shared_store=store, cache_key="m"
        )

    cached_a = build(model_a, DEVICE_A)
    cached_b = build(model_b, DEVICE_B)

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def run(cached, device):
        try:
            xd = x.to(device)
            for _ in range(20):
                barrier.wait()  # maximise overlap of the two devices' loads
                cached.full_load_to_vram()
                out = cached.model(xd)
                assert torch.allclose(out.cpu(), reference, atol=1e-5)
                cached.full_unload_from_vram()
        except Exception as e:  # noqa: BLE001 - surface to main thread
            errors.append(e)
            try:
                barrier.abort()
            except Exception:
                pass

    t_a = threading.Thread(target=run, args=(cached_a, DEVICE_A))
    t_b = threading.Thread(target=run, args=(cached_b, DEVICE_B))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert not errors, f"Concurrent load/unload corrupted results: {errors[0]!r}"
    # Canonical CPU weights survived and are still shared.
    assert store.refcount("m") == 2
    cached_a.release_shared_weights()
    cached_b.release_shared_weights()
    assert "m" not in store


class _GGUFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _build_gguf_model(seed: int) -> _GGUFModel:
    """A small model whose linear weight is a Q8_0 GGML-quantized (CPU-resident) tensor.

    This mirrors how large quantized transformers/encoders are stored: the weights live on the CPU
    as GGMLTensors and are dequantized on the fly during the forward pass. It is the path that goes
    through the shared-CPU-weights mechanism, so it validates that re-pointing a quantized state
    dict across devices preserves correct dequantized inference.
    """
    torch.manual_seed(seed)
    model = _GGUFModel()
    model.linear.weight = torch.nn.Parameter(quantize_tensor(model.linear.weight, gguf.GGMLQuantizationType.Q8_0))
    return model


@requires_two_gpus
def test_shared_gguf_quantized_weights_correct_on_both_gpus():
    """A GGUF-quantized model loaded on two GPUs from one shared CPU copy must dequantize and
    compute correct results on both devices."""
    x = torch.randn(1, 32, dtype=torch.float32)

    # Reference: a standalone copy of the same (seed-0) quantized weights, run via the autocast
    # custom layers. Weights stay on CPU; compute happens on the device.
    reference_model = _build_gguf_model(0)
    apply_custom_layers_to_model(reference_model, device_autocasting_enabled=True)
    reference = reference_model(x.to(DEVICE_A)).cpu()

    model_a = _build_gguf_model(0)
    model_b = _build_gguf_model(1)  # different weights; sharing must overwrite with canonical

    store = SharedCpuWeightsStore()
    # enable_partial_loading=True routes quantized nn.Modules through CachedModelWithPartialLoad.
    cache_a = _make_cache(store, DEVICE_A, partial=True)
    cache_b = _make_cache(store, DEVICE_B, partial=True)
    try:
        cache_a.put("m", model_a)
        ram_one_device = store.total_bytes_in_use()
        cache_b.put("m", model_b)

        # One shared CPU copy of the quantized weights; second device adds no RAM.
        assert store.refcount("m") == 2
        assert store.total_bytes_in_use() == ram_one_device
        rec_a = cache_a.get("m")
        rec_b = cache_b.get("m")
        assert rec_a.cached_model.get_cpu_state_dict() is rec_b.cached_model.get_cpu_state_dict()
        # model_b's quantized weight was re-pointed at model_a's canonical tensor.
        assert rec_b.cached_model.model.linear.weight.data_ptr() == rec_a.cached_model.model.linear.weight.data_ptr()

        cache_a.lock(rec_a, None)
        cache_b.lock(rec_b, None)
        try:
            out_a = rec_a.cached_model.model(x.to(DEVICE_A))
            out_b = rec_b.cached_model.model(x.to(DEVICE_B))
        finally:
            cache_a.unlock(rec_a)
            cache_b.unlock(rec_b)

        # Both GPUs reproduce the reference dequantized output.
        assert torch.allclose(out_a.cpu(), reference, atol=1e-5)
        assert torch.allclose(out_b.cpu(), reference, atol=1e-5)
    finally:
        cache_a.shutdown()
        cache_b.shutdown()
