import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from tests.backend.model_manager.load.model_cache.dummy_module import DummyModule

parameterize_mps_and_cuda = pytest.mark.parametrize(
    ("device"),
    [
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available.")
        ),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
    ],
)


@parameterize_mps_and_cuda
def test_cached_model_total_bytes(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available.")

    model = DummyModule()
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device(device))
    linear_numel = 10 * 10 + 10
    assert cached_model.total_bytes() == linear_numel * 4 * 2

    cached_model.model.to(dtype=torch.float16)
    assert cached_model.total_bytes() == linear_numel * 2 * 2


@parameterize_mps_and_cuda
def test_cached_model_cur_vram_bytes(device: str):
    model = DummyModule()
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device(device))
    assert cached_model.cur_vram_bytes() == 0

    cached_model.model.to(device=torch.device(device))
    assert cached_model.cur_vram_bytes() == cached_model.total_bytes()


@parameterize_mps_and_cuda
def test_cached_model_partial_load(device: str):
    model = DummyModule()
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device(device))
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    target_vram_bytes = int(model_total_bytes * 0.6)
    loaded_bytes = cached_model.partial_load_to_vram(target_vram_bytes)
    assert loaded_bytes > 0
    assert loaded_bytes < model_total_bytes
    assert loaded_bytes == cached_model.cur_vram_bytes()


@parameterize_mps_and_cuda
def test_cached_model_partial_unload(device: str):
    model = DummyModule()
    model.to(device=torch.device(device))
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device(device))
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == model_total_bytes

    bytes_to_free = int(model_total_bytes * 0.4)
    freed_bytes = cached_model.partial_unload_from_vram(bytes_to_free)
    assert freed_bytes >= bytes_to_free
    assert freed_bytes < model_total_bytes
    assert freed_bytes == model_total_bytes - cached_model.cur_vram_bytes()
