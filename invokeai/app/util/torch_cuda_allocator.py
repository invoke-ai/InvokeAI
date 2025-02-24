import logging
import os


def configure_torch_cuda_allocator(pytorch_cuda_alloc_conf: str, logger: logging.Logger | None = None):
    """Configure the PyTorch CUDA memory allocator. See
    https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf for supported
    configurations.
    """

    # Raise if the PYTORCH_CUDA_ALLOC_CONF environment variable is already set.
    prev_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)
    if prev_cuda_alloc_conf is not None:
        raise RuntimeError(
            f"Attempted to configure the PyTorch CUDA memory allocator, but PYTORCH_CUDA_ALLOC_CONF is already set to "
            f"'{prev_cuda_alloc_conf}'."
        )

    # Configure the PyTorch CUDA memory allocator.
    # NOTE: It is important that this happens before torch is imported.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf

    import torch

    # Relevant docs: https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attempted to configure the PyTorch CUDA memory allocator, but no CUDA devices are available."
        )

    # Verify that the torch allocator was properly configured.
    allocator_backend = torch.cuda.get_allocator_backend()
    expected_backend = "cudaMallocAsync" if "cudaMallocAsync" in pytorch_cuda_alloc_conf else "native"
    if allocator_backend != expected_backend:
        raise RuntimeError(
            f"Failed to configure the PyTorch CUDA memory allocator. Expected backend: '{expected_backend}', but got "
            f"'{allocator_backend}'. Verify that 1) the pytorch_cuda_alloc_conf is set correctly, and 2) that torch is "
            "not imported before calling configure_torch_cuda_allocator()."
        )

    if logger is not None:
        logger.info(f"PyTorch CUDA memory allocator: {torch.cuda.get_allocator_backend()}")
