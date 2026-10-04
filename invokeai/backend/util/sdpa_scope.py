"""One process-wide owner for torch's SDPA backend selection.

``torch.nn.attention.sdpa_kernel`` reads and writes ``at::globalContext()`` -- process-global state,
not thread-local, and torch offers no thread-scoped variant. Its save/restore is also not atomic
across threads. With two overlapping windows (A enters, B enters, A exits, B exits) B restores the
snapshot it took *inside* A's window, so A's values become permanent. Measured on the Krea-2 list:
the global priority order went from ``[FLASH, EFFICIENT, MATH, CUDNN]`` to
``[FLASH, CUDNN, EFFICIENT, MATH]`` and stayed there for the life of the process; with an exclusive
override in one thread, the other three backends were left *disabled* process-wide.

That is reachable in ordinary operation -- InvokeAI runs one generation session per GPU
concurrently, and both the Krea-2 attention processor and the Anima denoise loop open windows -- and
it leaks well past the architecture that opened it: every other model in the process then picks
kernels by the leaked order, and `backend.util.attention` sizes working memory from that same order.

So the globals get a single owner. The outermost window is the only one that writes them; a window
opened while another is already active runs under the active policy instead of overwriting it. Two
concurrent sessions cannot hold different values of a process-global setting -- that is torch's
constraint, not a choice made here -- but the state is now saved once and restored once, which is
what stops the corruption.

The cost of that choice is that a window can silently not apply. For the default ranking that only
decides which of four enabled kernels runs first. For an *exclusive* policy it is load-bearing: the
whole point of ``INVOKE_KREA2_SDPA_BACKEND=cudnn`` is that a run which completes proves that kernel
served it, and a window that did not apply voids the measurement. So non-owning entries are counted
(:func:`foreign_window_entries`) for a benchmark to report against, and one is logged.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

Policy = tuple[tuple[SDPBackend, ...], bool]

_lock = threading.Lock()
_depth = 0
_window: AbstractContextManager[object] | None = None
_policy: Policy | None = None
_foreign_entries = 0
_reported: set[tuple[Policy, Policy]] = set()


def foreign_window_entries() -> int:
    """How many windows have so far run under a policy other than the one they asked for.

    Monotonic for the life of the process. Read it before and after a measured run: a difference
    means the run did not have the backend selection it asked for.
    """
    return _foreign_entries


@contextmanager
def sdpa_policy(backends: Sequence[SDPBackend], set_priority: bool = True) -> Iterator[bool]:
    """Run the block with `backends` as torch's SDPA backend selection, if no other window owns it.

    A drop-in for ``sdpa_kernel(list(backends), set_priority=set_priority)`` that is safe to open
    from several threads. Entering while another window is open is a no-op for the duration of this
    block: the caller runs under the open policy, and the globals are restored once, by the window
    that set them.

    Yields whether this window owns the globals, so a caller whose correctness depends on its policy
    actually applying can tell.
    """
    global _depth, _window, _policy, _foreign_entries

    requested = (tuple(backends), set_priority)
    collision: tuple[Policy, Policy] | None = None
    with _lock:
        owned = _depth == 0
        if owned:
            # Published only once the enter has succeeded: a half-entered window must not be left
            # behind for the next caller to try to exit.
            window = sdpa_kernel(list(backends), set_priority=set_priority)
            window.__enter__()
            _window, _policy = window, requested
        elif requested != _policy:
            _foreign_entries += 1
            assert _policy is not None
            if (_policy, requested) not in _reported:
                _reported.add((_policy, requested))
                collision = (_policy, requested)
        _depth += 1

    # Outside the lock on purpose: `log_handlers` can be `http=` or `syslog=`, and a stalled handler
    # holding this lock would stall every session's attention path.
    if collision is not None:
        _report(*collision)

    try:
        yield owned
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _window is not None:
                window, _window, _policy = _window, None, None
                window.__exit__(None, None, None)


def _report(active: Policy, requested: Policy) -> None:
    # Once per distinct pair, not once per process: a benign default-vs-default collision must not
    # swallow the line for an exclusive policy that did not apply.
    exclusive = len(requested[0]) < len(active[0])
    message = (
        f"SDPA backend policy {_describe(requested)} was not applied: another session already holds "
        f"the window with {_describe(active)}. torch's backend selection is process-global, so two "
        "concurrent sessions cannot both set it."
    )
    if exclusive:
        logger.warning(f"{message} A run measured under this override did not get the kernel it asked for.")
    else:
        logger.info(message)


def _describe(policy: Policy) -> str:
    backends, set_priority = policy
    return f"[{', '.join(b.name for b in backends)}] set_priority={set_priority}"
