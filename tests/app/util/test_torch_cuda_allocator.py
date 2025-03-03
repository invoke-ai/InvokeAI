import pytest
import torch

from invokeai.app.util.torch_cuda_allocator import configure_torch_cuda_allocator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device.")
def test_configure_torch_cuda_allocator_raises_if_torch_is_already_imported():
    """Test that configure_torch_cuda_allocator() raises a RuntimeError if torch is already imported."""
    import torch  # noqa: F401

    with pytest.raises(RuntimeError, match="Failed to configure the PyTorch CUDA memory allocator."):
        configure_torch_cuda_allocator("backend:cudaMallocAsync")
