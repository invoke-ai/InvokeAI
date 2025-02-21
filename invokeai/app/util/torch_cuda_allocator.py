import os


def is_torch_cuda_malloc_enabled():
    """Check if the cudaMallocAsync memory allocator backend is being used."""
    # NOTE: We do not import torch at the file level, because enable_torch_cuda_malloc() must be called before torch is
    # imported.
    import torch

    if not torch.cuda.is_available():
        return False

    # Allocate something on a CUDA device so that there are memory stats to check.
    _ = torch.zeros(1, device="cuda")

    # Many of the memory stats are populated when using the native torch memory allocator, but fixed at 0 when using the
    # cudaMallocAsync memory allocator. The "active.all.allocated" stat is one that is not populated when using the
    # cudaMallocAsync memory allocator, so we can use it to chek if the cudaMallocAsync memory allocator is being used.
    return torch.cuda.memory_stats()["active.all.allocated"] == 0


def enable_torch_cuda_malloc():
    """Configure the PyTorch CUDA memory allocator to use the cudaMallocAsync memory allocator backend."""

    # Raise if the PYTORCH_CUDA_ALLOC_CONF environment variable is already set.
    prev_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if prev_cuda_alloc_conf is not None:
        raise RuntimeError(
            f"Attempted to configure the PyTorch CUDA memory allocator, but PYTORCH_CUDA_ALLOC_CONF is already set to "
            f"'{prev_cuda_alloc_conf}'."
        )

    # Enable the cudaMallocAsync memory allocator backend.
    # NOTE: It is important that this happens before torch is imported.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    import torch

    # Relevant docs: https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attempted to configure the PyTorch CUDA memory allocator, but no CUDA devices are available."
        )

    # Confirm that the cudaMallocAsync memory allocator backend is now being used.
    if not is_torch_cuda_malloc_enabled():
        raise RuntimeError(
            "Failed to enable the cudaMallocAsync memory allocator backend. This likely means that the torch memory "
            "allocator was initialized before calling this function."
        )
