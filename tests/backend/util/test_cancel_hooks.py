"""Tests for `cancel_before_forward`: CPU behavior on a small module chain, plus one slow GPU test."""

import threading
import time

import pytest
import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.util.cancel_hooks import cancel_before_forward

CPU = torch.device("cpu")


class _Chain(torch.nn.Module):
    """Three units of work run in sequence, each recording that it ran."""

    def __init__(self) -> None:
        super().__init__()
        self.units = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.ran: list[int] = []
        for index, unit in enumerate(self.units):
            unit.register_forward_hook(lambda module, args, output, index=index: self.ran.append(index))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            x = unit(x)
        return x


def _no_cancel_hooks(chain: _Chain) -> bool:
    return all(not unit._forward_pre_hooks for unit in chain.units)


def test_cancel_stops_at_the_next_unit_and_removes_the_hooks():
    chain = _Chain()
    polls = iter([False, True])
    with pytest.raises(CanceledException):
        with cancel_before_forward(chain.units, lambda: next(polls), CPU):
            chain(torch.zeros(1, 4))
    assert chain.ran == [0], "the forward did not stop at the first unit boundary after the cancel"
    assert _no_cancel_hooks(chain)


def test_uncanceled_forward_polls_once_per_unit_and_removes_the_hooks():
    chain = _Chain()
    polls = 0

    def is_canceled() -> bool:
        nonlocal polls
        polls += 1
        return False

    with cancel_before_forward(chain.units, is_canceled, CPU):
        chain(torch.zeros(1, 4))
        assert not _no_cancel_hooks(chain), "hooks must stay installed for the whole context"
    assert chain.ran == [0, 1, 2]
    assert polls == 3
    assert _no_cancel_hooks(chain)


def test_hooks_are_removed_when_the_forward_raises_something_else():
    chain = _Chain()
    with pytest.raises(RuntimeError):
        with cancel_before_forward(chain.units, lambda: False, CPU):
            chain(torch.zeros(1, 5))  # wrong width: the first unit raises
    assert _no_cancel_hooks(chain)


def test_no_callback_installs_nothing():
    chain = _Chain()
    with cancel_before_forward(chain.units, None, CPU):
        assert _no_cancel_hooks(chain)
        chain(torch.zeros(1, 4))
    assert chain.ran == [0, 1, 2]


class _HeavyBlock(torch.nn.Module):
    """Tens of milliseconds of GPU work per call: a chain of them outlasts the CPU's enqueue time
    by orders of magnitude, as a real transformer step or VAE decode does."""

    DIM = 8192

    def __init__(self, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(self.DIM, self.DIM, dtype=torch.bfloat16, device=device) * 0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for _ in range(8):
            hidden_states = torch.nn.functional.layer_norm(hidden_states @ self.weight, (self.DIM,))
        return hidden_states


class _HeavyStack(torch.nn.Module):
    def __init__(self, device: torch.device, num_blocks: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_HeavyBlock(device) for _ in range(num_blocks)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures cancel latency against queued GPU kernels")
def test_mid_call_cancel_idles_the_gpu_within_a_unit():
    """Kernel launches are asynchronous: without a wait in the poll, the CPU enqueues every unit of
    a call in tens of milliseconds and parks at the first host sync, so a cancel landing mid-call
    is not seen until the call returns and the GPU grinds through all of it (observed on the
    dual-GPU rig: a cancel during H3 denoise step 2 kept the GPU busy until the step ended). The
    poll must wait for the queued work, so that the GPU goes idle within about a unit of the cancel."""
    device = torch.device("cuda")
    num_blocks = 16
    torch.manual_seed(0)
    stack = _HeavyStack(device, num_blocks).eval()
    hidden_states = torch.randn(4096, _HeavyBlock.DIM, dtype=torch.bfloat16, device=device)
    cancel = threading.Event()

    def timed_forward(stack: _HeavyStack, hidden_states: torch.Tensor) -> tuple[float, bool]:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        canceled = False
        try:
            with torch.no_grad(), cancel_before_forward(stack.blocks, cancel.is_set, device):
                stack(hidden_states)
        except CanceledException:
            canceled = True
        torch.cuda.synchronize(device)
        return time.perf_counter() - start, canceled

    full_call, canceled = timed_forward(stack, hidden_states)
    assert not canceled
    block_time = full_call / num_blocks

    cancel_at = full_call * 0.4
    timer = threading.Timer(cancel_at, cancel.set)
    timer.start()
    try:
        elapsed, canceled = timed_forward(stack, hidden_states)
    finally:
        timer.cancel()
        # 2 GiB of synthetic weights: hand them back rather than leaving them cached for the session.
        del stack, hidden_states
        torch.cuda.empty_cache()
    assert canceled, f"cancel at {cancel_at:.2f}s was never seen inside the {full_call:.2f}s call"
    idle_after_cancel = elapsed - cancel_at
    # The bound is one block plus the poll's own latency; without the wait it is the rest of the
    # call (~0.6 * full_call). Allow generous scheduling jitter on the dev machine.
    assert idle_after_cancel < 3 * block_time, (
        f"GPU stayed busy {idle_after_cancel:.2f}s after the cancel (block ~{block_time:.2f}s, call {full_call:.2f}s)"
    )
