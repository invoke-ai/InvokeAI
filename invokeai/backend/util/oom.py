"""Recognising an out-of-memory failure across backends."""

import torch


def is_oom_error(e: RuntimeError) -> bool:
    """Return True if the error indicates an out-of-memory condition.

    The caching allocator raises torch.cuda.OutOfMemoryError, but an OOM surfaced from inside a
    cuDNN/cuBLAS kernel (e.g. workspace allocation in a VAE's convolutions) arrives as a plain
    RuntimeError, which must be matched by message. XPU exhaustion likewise arrives as a plain
    RuntimeError, naming the Level Zero/UR result code (`..._OUT_OF_DEVICE_MEMORY`) rather than the
    words "out of memory" -- so it needs its own spelling to be matched here.

    `out_of_host_memory` is knowingly over-broad: Level Zero returns it for driver-side resource
    failures generally (kernel compilation, handle exhaustion), not only host allocation. Matching
    it means a genuinely broken decode costs one wasted tiled retry before the error re-raises
    unchanged. That is preferred over the alternative -- a real host-memory exhaustion that skips
    the retry -- because the retry is bounded and non-destructive, while a missed OOM fails a
    generation that would have succeeded tiled.

    This lives in one place on purpose: a second copy is how the next backend's spelling gets added
    to one of them only.
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return (
        "out of memory" in msg
        or "out_of_device_memory" in msg
        or "out_of_host_memory" in msg
        or "cudnn_status_alloc_failed" in msg
        or "cublas_status_alloc_failed" in msg
    )
