"""Cancel a session from inside a long model call, at the boundaries of its units of work."""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException


@contextmanager
def cancel_before_forward(
    modules: Iterable[torch.nn.Module], is_canceled: Callable[[], bool] | None, device: torch.device
) -> Iterator[None]:
    """Poll ``is_canceled`` before every forward of ``modules`` for the duration of the context.

    A long inference call — a denoise step through 50 transformer blocks, a VAE decode over a few
    hundred tiles — returns to the caller only once, so a cancel polled between calls leaves the
    GPU busy for the rest of the call. Polling from a forward pre-hook on each unit of work bounds
    the latency to one unit — provided the poll waits for the queued kernels first. Kernel launches
    are asynchronous: left alone, the CPU enqueues the whole call's kernels in tens of milliseconds
    and parks at the first host sync, so every poll would already have run by the time a mid-call
    cancel arrived and the GPU would grind through regardless. Waiting for the previous unit's
    kernels before each poll costs one launch gap, under a millisecond (below 0.5% of a ~1 s
    transformer block at production sizes).

    A cancel raises ``CanceledException`` from inside the forward. The hooks are removed on exit:
    the modules belong to a shared cache resident, and the callback belongs to this session.
    """
    if is_canceled is None:
        yield
        return

    def raise_if_canceled(module: torch.nn.Module, args: tuple[object, ...]) -> None:
        if device.type != "cpu":
            torch.accelerator.synchronize(device)
        if is_canceled():
            raise CanceledException

    handles = [module.register_forward_pre_hook(raise_if_canceled) for module in modules]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
