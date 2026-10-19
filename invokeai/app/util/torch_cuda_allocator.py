import importlib.metadata
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

# Every variable torch reads its caching-allocator configuration from. `ModelCache._expandable_segments_enabled` keeps
# its own copy of this list: the model cache imports torch, and this module must not.
_ALLOCATOR_CONF_ENV_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")

ROCM_WINDOWS_ALLOC_CONF = "expandable_segments:True"


def _installed_torch_version() -> str | None:
    """The installed torch distribution's version, read from its metadata so torch itself is not imported."""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_torch_is_rocm() -> bool:
    """Whether the installed torch is a ROCm build, without importing torch.

    AMD's published wheels carry `+rocm` in the distribution version, but a wheel built locally is versioned like
    `2.8.0a0+gitfc14c65`. Both record a `hip` version in torch's own `version.py`, which is read as a file -- the same
    value `torch.version.hip` returns, which is what `invokeai.backend.util.wddm` gates on. Keeping the two answers
    together matters: a build that gets the video-memory budget but not this allocator default fragments its way into
    system memory with nothing in the log to say why.
    """
    version = _installed_torch_version()
    if version is not None and "+rocm" in version:
        return True
    try:
        spec = importlib.util.find_spec("torch")  # locates the package; does not execute it
        locations = list(spec.submodule_search_locations or []) if spec is not None else []
        for location in locations:
            source = Path(location, "version.py")
            if source.is_file():
                return (
                    re.search(r"^hip\s*(?::[^=]+)?=\s*['\"]", source.read_text(encoding="utf-8"), re.MULTILINE)
                    is not None
                )
    except Exception:
        return False
    return False


def apply_rocm_windows_allocator_default(logger: logging.Logger) -> None:
    """Default a ROCm build on Windows to the expandable-segments allocator, unless the allocator is configured already.

    Windows does not fail an allocation it cannot place in contiguous VRAM: it puts all of it in shared system memory,
    silently and for as long as the caching allocator keeps the segment. Freed-up VRAM does not pull it back, so one
    fragmented allocation keeps slowing every later tensor that reuses the segment. Measured on an RX 9060 XT with
    torch 2.12+rocm7.14: a 2 GiB tensor landed in system memory with 4.7 GiB of VRAM free in 256 MiB holes, and Z-Image
    at 1024px denoised in 144/139/186 s over three runs, against 37/34/34 s with expandable segments, which map a large
    block from small physical pieces instead of needing one contiguous range.

    Must run before torch is imported. Any allocator variable already in the environment, like an explicit
    `pytorch_cuda_alloc_conf` (whose caller skips this default), wins.
    """
    if sys.platform != "win32" or any(os.environ.get(var) for var in _ALLOCATOR_CONF_ENV_VARS):
        return
    if not _installed_torch_is_rocm():
        return
    if "torch" in sys.modules:
        # Setting the variable now would change nothing but the model cache's reading of it.
        logger.warning(
            "ROCm on Windows: torch was imported before the allocator default could apply; leaving it unset."
        )
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ROCM_WINDOWS_ALLOC_CONF
    logger.info(
        f"ROCm on Windows: using the expandable-segments allocator (PYTORCH_CUDA_ALLOC_CONF={ROCM_WINDOWS_ALLOC_CONF}), "
        "so fragmented VRAM does not push allocations into shared system memory. Set pytorch_cuda_alloc_conf to "
        "override it, e.g. to 'expandable_segments:False'."
    )


def configure_torch_cuda_allocator(pytorch_cuda_alloc_conf: str, logger: logging.Logger):
    """Configure the PyTorch CUDA memory allocator. See
    https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf for supported
    configurations.
    """

    if "torch" in sys.modules:
        raise RuntimeError("configure_torch_cuda_allocator() must be called before importing torch.")

    # Log a warning if the PYTORCH_CUDA_ALLOC_CONF environment variable is already set.
    prev_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)
    if prev_cuda_alloc_conf is not None:
        if prev_cuda_alloc_conf == pytorch_cuda_alloc_conf:
            logger.info(
                f"PYTORCH_CUDA_ALLOC_CONF is already set to '{pytorch_cuda_alloc_conf}'. Skipping configuration."
            )
            return
        else:
            logger.warning(
                f"Attempted to configure the PyTorch CUDA memory allocator with '{pytorch_cuda_alloc_conf}', but PYTORCH_CUDA_ALLOC_CONF is already set to "
                f"'{prev_cuda_alloc_conf}'. Skipping configuration."
            )
            return

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

    logger.info(f"PyTorch CUDA memory allocator: {torch.cuda.get_allocator_backend()}")
