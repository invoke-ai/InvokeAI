"""One process-wide owner for torch's SDPA backend selection: the kernel priority order, and the per-backend enable
flags while an exclusive window is open.

**The order.** :data:`DEFAULT_SDPA_PRIORITY` is the baseline, written into torch once at startup by
:func:`install_default_sdpa_priority`, before any generation runs. It is one order for the whole process rather than
one per architecture because torch's selection is process-global: per-architecture orders cannot hold while two GPUs
run sessions concurrently, and no measured architecture wants a different one. The server admin can pin one kernel
(``attention_backend: cudnn | flash | efficient | math``, named as in :data:`SDPA_KERNELS`): it is then ranked first,
ahead of the rest of the baseline in its order. A pinned kernel is a ranking too, never an exclusive choice, so a call
it cannot serve runs on the next kernel; `backend.util.attention.install_pinned_sdpa_kernel_report` logs those calls.
Windows never touch the order. One thing changes it afterwards: when an attention call fails inside cuDNN at run time
-- torch checks cuDNN's eligibility statically, so a plan the cuDNN frontend cannot build, an old cuDNN library swapped
in by hand or a GPU generation it mishandles raises instead of falling back -- the SDPA wrapper from
`backend.util.attention.install_cudnn_sdpa_fallback` calls :func:`demote_cudnn_attention`, which ranks cuDNN behind math
for the rest of the process, pinned or not, and retries the call on the next kernel.

The dispatcher honours the order for CUDA and ROCm tensors. On the CPU torch picks flash wherever it can and math
otherwise, whatever the order says (measured on torch 2.13).

**The flags.** ``torch.nn.attention.sdpa_kernel`` reads and writes ``at::globalContext()`` -- process-global state,
not thread-local, and torch offers no thread-scoped variant. Its save/restore is also not atomic across threads. With
two overlapping windows (A enters, B enters, A exits, B exits) B restores the snapshot it took *inside* A's window, so
A's values become permanent: measured, an exclusive override left the other backends *disabled* process-wide. Every
other model in the process then picks kernels by the leaked flags, and `backend.util.attention` sizes working memory
from them. That is reachable whenever two sessions run under the ``INVOKE_KREA2_SDPA_BACKEND`` override at once.

So windows get a single owner, :func:`sdpa_policy`. The outermost window is the only one that writes the flags; a
window opened while another is open runs under that one, and the flags are restored once, by the window that set
them. The override's choice is fixed for the process, so nested windows ask for what is already in force.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from functools import lru_cache

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

# The process-wide SDPA priority order, ranked ahead of anything else torch lists (OVERRIDEABLE today), which keeps
# torch's relative order behind it. torch's own order is [FLASH, EFFICIENT, MATH, CUDNN, ...]: cuDNN ranks behind the
# always-eligible math kernel, so it is practically never chosen.
#
# Flash first: where the build has it (Linux CUDA) it is the fastest kernel and already what torch's order picks, so
# there this order only changes the calls flash refuses -- any attention mask, such as the regional-prompting blocks
# pass, or heads wider than 256 -- which cuDNN now takes instead of memory-efficient. Measured on the Krea-2 attention
# shape ([1, 48, 4608, 128], bf16), per call:
#
#                     RTX 4090 / Windows      RTX 30-series / Linux
#   flash             not compiled in         19.74 ms
#   cudnn              3.72 ms                21.27 ms
#   efficient          5.92 ms                31.45 ms
#   math              51.23 ms               168.48 ms
#
# cuDNN second: Windows CUDA builds have no flash, so there cuDNN serves every call it accepts. Denoise time per
# generation on the RTX 4090 under Windows, against torch's order: FLUX.1 -7 % / -17 % (1024 / 2048 px), FLUX.2 Klein
# -7 / -14, Z-Image -7 / -13, Qwen-Image -4 / -14, SDXL +/-0 / -8, LTX-2 video -10, Wan 2.2 5B -18, Wan 2.2 A14B -16,
# Anima about -6 per step at 1024; peak VRAM unchanged. Images at the same seed change slightly, as with any kernel
# swap (bf16 accumulation order).
#
# Every entry is a fallback, never an exclusive choice: the dispatcher skips a kernel that cannot serve a call, so the
# order degrades on its own. ROCm builds have no cuDNN attention, so there it selects exactly what torch's order did.
DEFAULT_SDPA_PRIORITY = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)

# The kernels by the names a user writes: `attention_backend` pins one of them, `INVOKE_KREA2_SDPA_BACKEND` opens an
# exclusive window on one.
SDPA_KERNELS: dict[str, SDPBackend] = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
}

# Serialises writes to the priority order: startup, and two GPU workers whose cuDNN calls fail at once.
_order_lock = threading.Lock()

_window_lock = threading.Lock()
_depth = 0
_window: AbstractContextManager[object] | None = None


def install_default_sdpa_priority(pinned: SDPBackend | None = None) -> bool:
    """Make :data:`DEFAULT_SDPA_PRIORITY` torch's SDPA priority order for the rest of the process, with `pinned`, when
    given, ranked ahead of it.

    Backends torch lists beyond the ranking keep their relative order behind it, and the per-backend
    enable flags are left alone. Idempotent. Writing the order creates no CUDA context.

    Uses the private calls ``sdpa_kernel`` itself makes. If a torch release renames, removes or changes
    them, this logs that, leaves torch's own order rather than stopping the server, and returns False.
    """
    ranking = DEFAULT_SDPA_PRIORITY if pinned is None else (pinned, *(b for b in DEFAULT_SDPA_PRIORITY if b != pinned))
    with _order_lock:
        try:
            ranked = [int(backend) for backend in ranking]
            order = torch._C._get_sdp_priority_order()
            torch._C._set_sdp_priority_order(ranked + [backend for backend in order if backend not in ranked])
        except Exception as error:
            logger.warning(
                f"Could not set the SDPA kernel priority order ({error!r}): torch's private priority-order API "
                "(torch._C._get_sdp_priority_order / _set_sdp_priority_order) is missing or has changed in this "
                "release. Attention keeps PyTorch's own order, which practically never picks the cuDNN kernel"
                + (", and attention_backend cannot rank a kernel first." if pinned is not None else ".")
            )
            return False
    return True


def demote_cudnn_attention() -> bool:
    """Rank cuDNN directly behind math for the rest of the process, if it ranks ahead of math now.

    Returns whether this call moved it, so of two GPU workers whose cuDNN calls fail at once exactly one
    does. The enable flags are left alone: an exclusive window restores them on exit, and the order is
    what is demoted. Nothing ranks cuDNN ahead again afterwards.
    """
    with _order_lock:
        try:
            order = list(torch._C._get_sdp_priority_order())
            cudnn, math = int(SDPBackend.CUDNN_ATTENTION), int(SDPBackend.MATH)
            if order.index(cudnn) > order.index(math):
                return False
            order.remove(cudnn)
            order.insert(order.index(math) + 1, cudnn)
            torch._C._set_sdp_priority_order(order)
        except Exception:
            return False
        return True


def cudnn_attention_demoted() -> bool:
    """Whether cuDNN ranks behind math now, as :func:`demote_cudnn_attention` leaves it."""
    try:
        order = list(torch._C._get_sdp_priority_order())
        return order.index(int(SDPBackend.CUDNN_ATTENTION)) > order.index(int(SDPBackend.MATH))
    except Exception:
        return False


def sdpa_priority_order() -> list[SDPBackend] | None:
    """torch's SDPA priority order as it stands now, or None where this torch cannot report it.

    Read per attention call where another kernel than flash ranks first (Krea-2's grouped-query decision) and inside
    an exclusive window (`backend.util.attention.sdpa_backend_choice`). torch's getter costs 0.07us, converting its
    answer to `SDPBackend`s 2.1us (measured on CPU), so the conversion is remembered for each order torch reports.
    """
    try:
        return list(_as_backends(tuple(torch._C._get_sdp_priority_order())))
    except Exception:
        return None


def sdpa_flash_ranks_first() -> bool | None:
    """Whether flash leads torch's SDPA priority order now, as it does unless `attention_backend` pins another kernel;
    None where this torch cannot report its order.

    Reads the first slot only, as an int: 0.08us, where reading the order and comparing an `SDPBackend` costs 0.65us
    (measured on CPU). Krea-2 asks it on every attention call.
    """
    try:
        return torch._C._get_sdp_priority_order()[0] == _FLASH
    except Exception:
        return None


_FLASH = int(SDPBackend.FLASH_ATTENTION)


@lru_cache(maxsize=8)
def _as_backends(order: tuple[int, ...]) -> tuple[SDPBackend, ...]:
    return tuple(SDPBackend(backend) for backend in order)


def describe_sdpa_priority_order() -> str:
    """torch's SDPA priority order for a log line, "unknown" where this torch cannot report it.

    Each backend is named once, where it first appears: torch 2.7 fills the last of its five slots with MATH again. A
    repeat is never reached, and naming it would read as a kernel tried twice.
    """
    order = sdpa_priority_order()
    return "unknown" if order is None else " > ".join(dict.fromkeys(backend.name for backend in order))


@contextmanager
def sdpa_policy(backends: Sequence[SDPBackend]) -> Iterator[None]:
    """Run the block with only `backends` enabled, unless another window is already open.

    A drop-in for ``sdpa_kernel(list(backends))`` that is safe to open from several threads: entering while another
    window is open runs the block under that one, and the flags are restored once, by the window that set them. Only
    the flags: the priority order is the process's, not a window's.
    """
    global _depth, _window

    with _window_lock:
        if _depth == 0:
            # Published only once the enter has succeeded: a half-entered window must not be left
            # behind for the next caller to try to exit.
            window = sdpa_kernel(list(backends))
            window.__enter__()
            _window = window
        _depth += 1
    try:
        yield
    finally:
        with _window_lock:
            _depth -= 1
            if _depth == 0 and _window is not None:
                window, _window = _window, None
                window.__exit__(None, None, None)
