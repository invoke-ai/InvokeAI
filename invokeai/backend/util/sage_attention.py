"""SageAttention for the attention inside diffusion models, opt-in via ``attention_backend: sage``.

SageAttention (thu-ml) quantizes Q/K to INT8 and P/V to FP8 or FP16 inside the attention kernel. On an RTX 4090 under
Windows it runs the attention shapes of the image and video transformers 2-3x faster than PyTorch's fastest SDPA kernel
there, and the gap widens with the token count. It is lossy: images differ from SDPA at the same seed (in a blind
comparison of 48 pairs over six architectures, none worse). That is why it is opt-in, and why it only runs where a
denoise invocation asks for it.

How it is wired:

* ``install_sage_attention`` rebinds ``torch.nn.functional.scaled_dot_product_attention`` once at startup, like
  ``install_rocm_sdpa_guard``. Every caller that resolves the function at call time inherits it: InvokeAI's own
  transformer code, diffusers' attention processors and diffusers' ``native`` dispatch backend. diffusers' own
  ``sage`` backend is not used: it raises on any mask, is process-global, and never sees the first two routes.
* The wrapper does nothing outside ``sage_attention_scope``. Text encoders, image encoders and VAEs therefore keep
  PyTorch's kernels, as do models whose denoise invocation has not opted in. The scope is a ``ContextVar``, so it is
  per thread: InvokeAI runs one generation session per GPU concurrently, and a process-global switch would leak
  between them (see ``sdpa_scope``).
* Inside the scope, a call goes to SageAttention only when ``_ineligible`` finds nothing wrong with it -- no mask,
  a head size the kernels run natively, half precision, sequences long enough to be worth the quantization.
  Everything else falls back to PyTorch with its arguments untouched.
"""

import functools
import inspect
import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.oom import is_oom_error

logger = InvokeAILogger.get_logger(__name__)

_SENTINEL = "_invokeai_sage_attention"
_MIN_VERSION = (2, 2, 0)
# Head sizes the kernels run natively. Other sizes are zero-padded by SageAttention (40 -> 64, 80 -> 128), which costs
# copies and measured slower than SDPA at the shapes that have them (SD1.5).
_HEAD_DIMS = frozenset({64, 128})
# Shortest query and key sequences SageAttention is used for. It quantizes Q, K and V in separate kernels first, a cost
# that grows with each sequence's length, while what it saves grows with their product -- so a short side on either
# end loses. Measured on an RTX 4090 against the faster of SDPA's default and cuDNN kernels: keys of 77 or 128 tokens
# run 0.2-0.8x (SDXL cross-attention, LTX-2's audio streams), 256 break even, 512 and longer run 1.4-1.9x; queries of
# 512 or 1024 tokens against 13.7k keys run 0.6x and 1.0x, 1536 and longer 1.4x and up.
_MIN_QUERY_LEN = 1536
_MIN_KEY_LEN = 512
_HALF_PRECISION = (torch.float16, torch.bfloat16)
# Largest relative L2 difference from PyTorch SDPA a device's first SageAttention result may have. Real attention
# stays well below it (worst measured: 0.19, in Qwen-Image's 53rd block); a kernel that never ran leaves uninitialized
# memory, which lands near 1 or is not finite.
_FIRST_CALL_MAX_ERROR = 0.5
# Query rows of the first call that are recomputed with PyTorch for that comparison, from the end so the last, partial
# tile is among them. A kernel that never ran garbles every row, so a slice catches what the whole output would; the
# whole output costs ~14 bytes per element to compare -- about 5 GB for one call of Wan 2.2 14B at 720p.
_FIRST_CALL_ROWS = 1024
# Triton failing to write a compiled kernel to its cache (on Windows, when two GPUs compile the same kernel at once)
# says nothing about the kernel: that call falls back and the next tries again. A device that fails this way this many
# times in a row is retired like any other, so a cache that can never be written does not recompile on every call.
_MAX_CACHE_FAILURES = 3

DOCS_URL = "https://invoke-ai.github.io/InvokeAI-7/configuration/optimization/sage-attention/"


@dataclass
class _Usage:
    """What one scope's attention calls ran on. Logged at DEBUG when the scope closes."""

    served: int = 0
    fell_back: Counter[str] = field(default_factory=Counter)
    # Set once SageAttention's quantized copies did not fit; the rest of the scope then stays on SDPA.
    memory_short: bool = False


_scope: ContextVar[_Usage | None] = ContextVar("invokeai_sage_attention_scope", default=None)
_lock = threading.Lock()
_disabled_devices: set[int] = set()
# (device, dtype, head size): the kernel variants whose first result was compared with SDPA.
_validated: set[tuple[int, torch.dtype, int]] = set()
_cache_failures: Counter[int] = Counter()
_announced_devices: set[int] = set()


@contextmanager
def sage_attention_scope() -> Iterator[None]:
    """Let SageAttention serve the eligible attention calls this thread makes inside the block.

    A no-op unless ``install_sage_attention`` installed the wrapper. Denoise invocations enter it around the
    transformer forward passes of architectures whose output was validated against SDPA.
    """
    usage = _Usage()
    token = _scope.set(usage)
    try:
        yield
    finally:
        _scope.reset(token)
        if usage.served or usage.fell_back:
            fallbacks = ", ".join(f"{reason}={count}" for reason, count in usage.fell_back.most_common())
            logger.debug(
                f"SageAttention: {usage.served + sum(usage.fell_back.values())} attention calls, "
                f"{usage.served} on SageAttention" + (f"; on PyTorch SDPA: {fallbacks}" if fallbacks else "")
            )


def _import_sageattention() -> tuple[str, Any]:
    """The only place that touches the optional package. Imported here, never at module import time."""
    from importlib.metadata import version

    import sageattention

    return version("sageattention"), sageattention


def _parse_version(version: str) -> tuple[int, int, int] | None:
    # Wheels carry local labels: "2.2.0+cu130torch2.10.0andhigher.post6".
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version)
    return (int(match[1]), int(match[2]), int(match[3] or 0)) if match else None


def _kernel(
    sageattention: Any, capability: tuple[int, int], cuda: tuple[int, int]
) -> Callable[..., torch.Tensor] | None:
    """SageAttention's kernel for a GPU of this compute capability, or None where it has none.

    The kernels and accumulation upstream SageAttention 2.2's ``sageattn`` (thu-ml) dispatches to: the FP16 CUDA
    kernel on sm80, the Triton one on sm86, nothing on sm87. The Windows build moved sm86 and sm87 to the CUDA kernel
    without saying why; neither has been measured here, so upstream's choice stands. From the Windows build come the
    sm100 kernel and the choice by the CUDA version torch was built for: the FP16 accumulation inside the FP8 kernels
    needs CUDA 12.8 to compile, and below that the build replaces it with a trap. Every kernel is called with
    ``smooth_k`` off, which ``sageattn`` always leaves on and has no argument to change.

    Smoothing subtracts K's mean over the sequence before quantizing; in the first block of Qwen-Image and Krea-2,
    where one channel of Q and K sits near +600 for every image token, that leaves the INT8 product Q.K no precision for
    the rest: cosine 0.18 and 0.53 against a float32 reference, 0.997 and 0.988 with smoothing off. Everywhere else
    measured it changes nothing, and off is 2-11 % faster (RTX 4090). Only sm_89 has been measured in InvokeAI; the
    sm86 Triton kernel was checked for accuracy on it.
    """
    major, minor = capability
    recent_cuda = cuda >= (12, 8)
    if (major, minor) == (8, 0):  # A100: FP16 P.V
        kernel, options = sageattention.sageattn_qk_int8_pv_fp16_cuda, {"pv_accum_dtype": "fp32"}
    elif (major, minor) == (8, 6):  # RTX 30-series, A10, A40: FP16 P.V in Triton
        kernel, options = sageattention.sageattn_qk_int8_pv_fp16_triton, {}
    elif (major, minor) == (8, 9):  # Ada: FP8 P.V
        kernel = sageattention.sageattn_qk_int8_pv_fp8_cuda
        options = {"pv_accum_dtype": "fp32+fp16" if recent_cuda else "fp32+fp32"}
    elif major == 9:  # Hopper
        kernel, options = sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90, {"pv_accum_dtype": "fp32+fp32"}
    elif (major, minor) in ((10, 0), (12, 0), (12, 1)):  # Blackwell
        kernel = sageattention.sageattn_qk_int8_pv_fp8_cuda
        options = {"qk_quant_gran": "per_warp", "pv_accum_dtype": "fp32+fp16" if recent_cuda else "fp32"}
    else:
        return None
    return functools.partial(kernel, smooth_k=False, **options)


def _unaccepted_arguments(kernel: functools.partial) -> list[str]:
    """Arguments this kernel would silently swallow: every SageAttention kernel ends in ``**kwargs``, so a renamed
    ``smooth_k`` would be ignored rather than rejected, and smoothing would quietly come back."""
    parameters = inspect.signature(kernel.func).parameters
    return [name for name in (*kernel.keywords, "tensor_layout", "is_causal", "sm_scale") if name not in parameters]


def _cuda_index(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> int | None:
    """The CUDA index of q/k/v when all three live on the same CUDA device, else None.

    SageAttention launches its kernels on the thread's current device without switching to the tensors' device, so the
    wrapper makes that device current for the call. Session workers pin their thread to their device already; a
    single-device install with `device: cuda:1` does not.
    """
    device = query.device
    if device.type != "cuda" or key.device != device or value.device != device:
        return None
    return device.index


def _ineligible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
    extra_arguments: bool,
) -> str | None:
    """Why this call must stay on PyTorch's SDPA, or None if SageAttention may serve it.

    Reads tensor metadata only: anything that synchronizes the device (inspecting a mask's values, say) would cost
    more than the kernel saves. Which device the call runs on is checked by the caller, which needs the index anyway.
    """
    if extra_arguments:
        return "arguments"
    if attn_mask is not None:
        return "mask"
    if dropout_p != 0.0:
        return "dropout"
    if is_causal:
        # No denoiser in scope attends causally, and torch and SageAttention have not been checked to align a causal
        # mask the same way when query and key lengths differ.
        return "causal"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or query.is_nested:
        return "shape"
    if query.dtype not in _HALF_PRECISION or key.dtype != query.dtype or value.dtype != query.dtype:
        return "dtype"
    batch, heads, query_len, head_dim = query.shape
    if head_dim not in _HEAD_DIMS or key.shape[-1] != head_dim or value.shape != key.shape:
        return "head_dim"
    kv_heads, key_len = key.shape[1], key.shape[2]
    if key.shape[0] != batch:
        return "shape"
    if kv_heads != heads and not (enable_gqa and heads % kv_heads == 0):
        return "heads"
    if query_len < _MIN_QUERY_LEN or key_len < _MIN_KEY_LEN:
        return "short"
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        return "layout"
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return "grad"
    if torch.is_autocast_enabled("cuda"):
        return "autocast"
    return None


def _check_first_result(
    index: int,
    original: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None,
    out: torch.Tensor,
) -> None:
    """Raise unless the first SageAttention result of this kernel variant agrees with PyTorch's.

    SageAttention launches its CUDA kernels without checking for errors, so a kernel the installed build lacks for
    this GPU raises nothing: it returns uninitialized memory, and the launch error surfaces in some later, unrelated
    torch operation -- outside the fallback. Recomputing a slice of the first result with PyTorch, and reading the
    comparison back to the host, brings both into the caller's ``try``. Once per device, precision and head size --
    the inputs that pick a different compiled kernel; later calls are not compared. The slice is the first sample's
    first K/V head group and its last ``_FIRST_CALL_ROWS`` query rows.
    """
    group = query.shape[1] // key.shape[1]
    rows = slice(-_FIRST_CALL_ROWS, None)
    expected = original(query[:1, :group, rows], key[:1, :1], value[:1, :1], scale=scale, enable_gqa=group > 1)
    difference = torch.linalg.vector_norm(out[:1, :group, rows] - expected, dtype=torch.float32)
    reference = torch.linalg.vector_norm(expected, dtype=torch.float32).clamp_min(1e-6)
    error = (difference / reference).item()
    if not math.isfinite(error) or error > _FIRST_CALL_MAX_ERROR:
        raise RuntimeError(f"its first result differs from PyTorch SDPA by {error:.2f} (relative L2)")
    with _lock:
        _validated.add((index, query.dtype, query.shape[-1]))


def _cache_failure(index: int) -> bool:
    """Count a Triton cache failure on this device; True once it failed too often in a row to keep retrying."""
    with _lock:
        _cache_failures[index] += 1
        return _cache_failures[index] >= _MAX_CACHE_FAILURES


def _disable(index: int, error: Exception, query: torch.Tensor, key: torch.Tensor) -> None:
    with _lock:
        first = index not in _disabled_devices
        _disabled_devices.add(index)
    if first:
        logger.warning(
            f"SageAttention failed on cuda:{index} for query {tuple(query.shape)}, key {tuple(key.shape)}, "
            f"{query.dtype} ({error!r}). Attention on this device uses PyTorch SDPA until restart; set "
            "`attention_backend: auto` to stop using SageAttention."
        )


def _announce(index: int, version: str, kernel: Callable[..., torch.Tensor]) -> None:
    if index in _announced_devices:
        return
    with _lock:
        if index in _announced_devices:
            return
        _announced_devices.add(index)
    major, minor = torch.cuda.get_device_capability(index)
    name = getattr(kernel, "func", kernel).__name__
    logger.info(
        f"SageAttention {version} serves diffusion-model attention on cuda:{index} "
        f"({torch.cuda.get_device_name(index)}, sm_{major}{minor}, {name})."
    )


def install_sage_attention() -> None:
    """Rebind ``F.scaled_dot_product_attention`` so ``sage_attention_scope`` can route calls to SageAttention.

    Idempotent and never raises: a machine that cannot run SageAttention logs why and keeps PyTorch's SDPA. Never
    installs on ROCm, which also keeps ``rocm_sdpa_chunks_math`` reading the ROCm guard's sentinel on the live binding.
    Installed once at startup via ``apply_monkeypatches`` when ``attention_backend`` is ``sage``.
    """
    from importlib.metadata import PackageNotFoundError

    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, _SENTINEL, False):
        return
    unavailable = "attention_backend is 'sage', but SageAttention is unavailable here: "
    see_docs = f"See {DOCS_URL}. Using PyTorch SDPA."
    try:
        if torch.version.hip is not None or torch.version.cuda is None or not torch.cuda.is_available():
            logger.warning(unavailable + "it needs an NVIDIA GPU with CUDA. Using PyTorch SDPA.")
            return
        capabilities = {index: torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())}
        if not any(capability >= (8, 0) for capability in capabilities.values()):
            logger.warning(
                unavailable + "it needs an NVIDIA GPU with compute capability 8.0 or newer. Using PyTorch SDPA."
            )
            return
        version, sageattention = _import_sageattention()
        parsed = _parse_version(version)
        if parsed is None or parsed < _MIN_VERSION:
            logger.warning(
                unavailable + f"version {version} is installed, and {'.'.join(map(str, _MIN_VERSION))} or newer is "
                f"required (the `sageattention` release on PyPI is SageAttention 1). {see_docs}"
            )
            return
        cuda = _parse_version(torch.version.cuda) or (0, 0, 0)
        kernels = {
            index: kernel for index, cap in capabilities.items() if (kernel := _kernel(sageattention, cap, cuda[:2]))
        }
        for kernel in kernels.values():
            if unaccepted := _unaccepted_arguments(kernel):
                logger.warning(
                    unavailable + f"its {kernel.func.__name__} does not take {', '.join(unaccepted)}, so this "
                    f"SageAttention {version} differs from the version InvokeAI was validated with. {see_docs}"
                )
                return
    except PackageNotFoundError:
        logger.warning(
            unavailable + f"`sageattention` imports, but has no package metadata to check its version. {see_docs}"
        )
        return
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "triton":
            hint = "it needs Triton, which on Windows is the `triton-windows` package"
        else:
            hint = "the `sageattention` package is not installed"
        logger.warning(unavailable + f"{hint}. {see_docs}")
        return
    except Exception as e:
        logger.warning(unavailable + f"loading it failed ({e!r}). {see_docs}")
        return

    original = functional.scaled_dot_product_attention

    @functools.wraps(original)
    def sage_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *args: Any,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        usage = _scope.get()
        if usage is not None:
            reason = _ineligible(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa, bool(args or kwargs))
            if reason is None and usage.memory_short:
                reason = "memory"
            if reason is None:
                index = _cuda_index(query, key, value)
                kernel = kernels.get(index) if index is not None and index not in _disabled_devices else None
                if index is None or kernel is None:
                    reason = "device"
                else:
                    try:
                        with torch.cuda.device(index):
                            out = kernel(
                                query,
                                key,
                                value,
                                tensor_layout="HND",
                                is_causal=False,
                                sm_scale=scale if scale is not None else query.shape[-1] ** -0.5,
                            )
                            if (index, query.dtype, query.shape[-1]) not in _validated:
                                _check_first_result(index, original, query, key, value, scale, out)
                    except Exception as e:
                        if isinstance(e, RuntimeError) and is_oom_error(e):
                            # SageAttention's quantized copies did not fit. SDPA needs less; the rest of this scope
                            # uses it rather than flushing the allocator's cache on every call, and the next
                            # generation tries SageAttention again. A CUDA fault that poisons the context is no OOM:
                            # it retires the device below, and SDPA then fails the same way.
                            usage.memory_short = True
                            reason = "memory"
                            logger.info(
                                f"SageAttention ran out of memory on cuda:{index} for query {tuple(query.shape)}; "
                                "the rest of this generation uses PyTorch SDPA."
                            )
                        elif isinstance(e, OSError) and not _cache_failure(index):
                            reason = "cache"
                        else:
                            _disable(index, e, query, key)
                            reason = "error"
                    else:
                        if _cache_failures[index]:
                            with _lock:
                                _cache_failures.pop(index, None)
                        usage.served += 1
                        _announce(index, version, kernel)
                        return out
            usage.fell_back[reason] += 1
        return original(
            query, key, value, attn_mask, dropout_p, is_causal, *args, scale=scale, enable_gqa=enable_gqa, **kwargs
        )

    setattr(sage_scaled_dot_product_attention, _SENTINEL, True)
    functional.scaled_dot_product_attention = sage_scaled_dot_product_attention
    unserved = sorted(index for index in capabilities if index not in kernels)
    logger.info(
        f"SageAttention {version} enabled for the attention inside supported diffusion models "
        "(attention_backend: sage). Text encoders, VAEs and masked attention keep PyTorch SDPA"
        + (f"; no SageAttention kernel for cuda:{', cuda:'.join(map(str, unserved))}." if unserved else ".")
    )
