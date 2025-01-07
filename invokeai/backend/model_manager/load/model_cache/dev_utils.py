from contextlib import contextmanager

import torch

from invokeai.backend.util.logging import InvokeAILogger


@contextmanager
def log_operation_vram_usage(operation_name: str):
    """A helper function for tuning working memory requirements for memory-intensive ops.

    Sample usage:

    ```python
    with log_operation_vram_usage("some_operation"):
        some_operation()
    ```
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    max_allocated_before = torch.cuda.max_memory_allocated()
    max_reserved_before = torch.cuda.max_memory_reserved()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        max_allocated_after = torch.cuda.max_memory_allocated()
        max_reserved_after = torch.cuda.max_memory_reserved()
        logger = InvokeAILogger.get_logger()
        logger.info(
            f">>>{operation_name} Peak VRAM allocated: {(max_allocated_after - max_allocated_before) / 2**20} MB, "
            f"Peak VRAM reserved: {(max_reserved_after - max_reserved_before) / 2**20} MB"
        )
