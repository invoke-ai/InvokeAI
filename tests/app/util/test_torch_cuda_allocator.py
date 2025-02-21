import pytest

from invokeai.app.util.torch_cuda_allocator import enable_torch_cuda_malloc, is_torch_cuda_malloc_enabled


def test_is_torch_cuda_malloc_enabled():
    """Test that if torch CUDA malloc hasn't been explicitly enabled, then is_torch_cuda_malloc_enabled() returns
    False.
    """
    assert not is_torch_cuda_malloc_enabled()


def test_enable_torch_cuda_malloc_raises_if_torch_is_already_imported():
    """Test that enable_torch_cuda_malloc() raises a RuntimeError if torch is already imported."""
    import torch  # noqa: F401

    with pytest.raises(RuntimeError):
        enable_torch_cuda_malloc()
