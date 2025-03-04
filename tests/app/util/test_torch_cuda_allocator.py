from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.util.torch_cuda_allocator import configure_torch_cuda_allocator
from tests.env_var_utils import set_env_var, unset_env_var


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device.")
def test_configure_torch_cuda_allocator_raises_if_torch_is_already_imported():
    """Test that configure_torch_cuda_allocator() raises a RuntimeError if torch is already imported."""

    with unset_env_var("PYTORCH_CUDA_ALLOC_CONF"):
        import torch  # noqa: F401

        mock_logger = MagicMock()
        with pytest.raises(RuntimeError, match="Failed to configure the PyTorch CUDA memory allocator."):
            configure_torch_cuda_allocator("backend:cudaMallocAsync", logger=mock_logger)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device.")
def test_configure_torch_cuda_allocator_warns_if_env_var_is_already_set():
    """Test that configure_torch_cuda_allocator() logs a warning if PYTORCH_CUDA_ALLOC_CONF is already set."""

    with set_env_var("PYTORCH_CUDA_ALLOC_CONF", "backend:native"):
        mock_logger = MagicMock()
        configure_torch_cuda_allocator("backend:cudaMallocAsync", logger=mock_logger)
        mock_logger.warning.assert_called_once()
        args, _kwargs = mock_logger.warning.call_args
        assert "PYTORCH_CUDA_ALLOC_CONF is already set" in args[0]
