import ctypes
import logging
import mimetypes
import platform
import re
import socket
import sys
from pathlib import Path

import torch

from invokeai.app.services.config.config_default import ATTENTION_BACKEND
from invokeai.backend.util.sdpa_scope import SDPA_KERNELS, describe_sdpa_priority_order, install_default_sdpa_priority


def find_open_port(port: int) -> int:
    """Find a port not in use starting at given port"""
    # Taken from https://waylonwalker.com/python-find-available-port/, thanks Waylon!
    # https://github.com/WaylonWalker
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            return find_open_port(port=port + 1)
        else:
            return port


def describe_torch_build() -> str:
    """One line naming the torch build and host platform, for the startup log.

    Out-of-tree torch builds (NVIDIA's `+cu134` wheels for Windows on ARM64) and ROCm builds all report
    `device.type == "cuda"`, so the device line alone does not say what is actually running; a bug
    report needs the version string, the accelerator API and the CPU architecture.
    """
    if torch.version.hip is not None:
        accelerator = f"HIP {torch.version.hip}"
    elif torch.version.cuda is not None:
        accelerator = f"CUDA {torch.version.cuda}"
    else:
        accelerator = "no CUDA/HIP"
    return f"torch {torch.__version__} ({accelerator}) on {platform.system()} {platform.machine()}"


def check_cudnn(logger: logging.Logger) -> None:
    """Check for cuDNN issues that could be causing degraded performance."""
    if torch.backends.cudnn.is_available():
        try:
            # Note: At the time of writing (torch 2.2.1), torch.backends.cudnn.version() only raises an error the first
            # time it is called. Subsequent calls will return the version number without complaining about a mismatch.
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"cuDNN version: {cudnn_version}")
        except RuntimeError as e:
            logger.warning(
                "Encountered a cuDNN version issue. This may result in degraded performance. This issue is usually "
                "caused by an incompatible cuDNN version installed in your python environment, or on the host "
                f"system. Full error message:\n{e}"
            )


def nvidia_driver_cuda_version() -> int | None:
    """The CUDA version the installed NVIDIA driver supports (12040 for 12.4), or None when there is no driver."""
    try:
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    except OSError:
        return None
    version = ctypes.c_int()
    try:
        if driver.cuDriverGetVersion(ctypes.byref(version)) != 0:
            return None
    except AttributeError:  # a stub library without the driver API
        return None
    return version.value


def check_cuda_build_compatibility(logger: logging.Logger) -> None:
    """Name the two ways a CUDA build of torch fails on an NVIDIA machine that torch itself only warns about.

    A driver too old for the build's CUDA version leaves torch without CUDA, so Invoke silently runs on the CPU and
    generation looks hung. A GPU older than the build's lowest architecture passes `is_available()` and then fails
    at the first kernel launch.
    """
    if torch.version.cuda is None or torch.version.hip is not None:
        return
    if not torch.cuda.is_available():
        driver = nvidia_driver_cuda_version()
        major, minor = (int(part) for part in torch.version.cuda.split(".")[:2])
        if driver is not None and driver < major * 1000 + minor * 10:
            logger.error(
                f"PyTorch here is built for CUDA {torch.version.cuda}, but the NVIDIA driver only supports CUDA "
                f"{driver // 1000}.{driver % 1000 // 10}, so Invoke is running on the CPU. Update the NVIDIA driver "
                "(R580 or newer for CUDA 13)."
            )
        return
    # Custom builds may list suffixed targets (sm_90a, sm_100f); a diagnostic must not stop the server over them.
    architectures = [int(m.group(1)) for arch in torch.cuda.get_arch_list() if (m := re.match(r"sm_(\d+)", arch))]
    if not architectures:
        return
    lowest = min(architectures)
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        if major * 10 + minor < lowest:
            logger.error(
                f"{torch.cuda.get_device_name(index)} has compute capability {major}.{minor}, but this PyTorch build "
                f"supports {lowest // 10}.{lowest % 10} and newer, so generation on it will fail. See the system "
                "requirements for the options on older GPUs."
            )


# The shape the diffusion transformers actually attend over: one image, many heads, head_dim 128.
# Availability can depend on the shape, so probing with a toy one would answer a different question.
_PROBE_HEADS = 24
_PROBE_SEQ_LEN = 1024
_PROBE_HEAD_DIM = 128


def probe_attention_backends(device: torch.device) -> dict[str, bool] | None:
    """Which fused SDPA backends this build and device can actually use.

    Availability is not a property of the torch version alone: ROCm builds have no cuDNN attention,
    Windows CUDA builds usually have no flash, and both depend on the device architecture. Returns
    None where the question does not apply (no CUDA/ROCm device) or cannot be answered.

    This asks about the *unmasked* case. A backend listed here can still be rejected for a specific
    call -- flash refuses the additive padding mask the regional-prompting paths pass, for instance --
    so this is a diagnostic, not a dispatch table.

    It allocates on `device`, which creates that device's CUDA primary context if nothing has yet.
    That context's VRAM is held for the process lifetime and `empty_cache()` does not release it, so
    this belongs on a path that was going to touch the GPU anyway -- not on the boot path, where it
    would park a context on an idle server, and on `cuda:0` rather than on a configured
    `generation_devices` entry.
    """
    if device.type != "cuda":
        return None
    try:
        probe = params = None
        try:
            probe = torch.empty(
                1,
                _PROBE_HEADS,
                _PROBE_SEQ_LEN,
                _PROBE_HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )
            params = torch.backends.cuda.SDPAParams(probe, probe, probe, None, 0.0, False, False)
            return {
                "cudnn": bool(torch.backends.cuda.can_use_cudnn_attention(params)),
                "flash": bool(torch.backends.cuda.can_use_flash_attention(params)),
                "efficient": bool(torch.backends.cuda.can_use_efficient_attention(params)),
                # Always present -- it is the unfused fallback, not a kernel that can be missing.
                "math": True,
            }
        finally:
            # Do not leave the probe tensor sitting in the caching allocator for the first
            # generation. `empty_cache()` only releases blocks nothing references. Both locals still
            # hold the probe here -- `SDPAParams` keeps a reference of its own -- so dropping both is
            # what makes the call do anything at all. Measured: without this the segment stays
            # reserved for the process lifetime, which is the opposite of what the line above says.
            #
            # Nested inside the `except` below, not beside it: a `finally` on the same `try` is not
            # covered by that `try`'s handler, so a raise from `empty_cache()` would escape a
            # function whose whole contract is that it does not.
            del probe, params
            torch.cuda.empty_cache()
    except Exception:
        # A diagnostic must never be the reason the server does not start.
        return None


# Devices already probed, whatever the outcome. The line answers a support question ("which kernels
# does this build have?"), so it is worth exactly once per device per process and nothing after that --
# and a probe that failed must not allocate and flush the allocator again on every later item.
_probed_attention_devices: set[str] = set()


def log_attention_backends(logger: logging.Logger, device: torch.device) -> None:
    """Log this device's SDPA backend availability and the priority order torch picks by, once per device.

    The point is support: a question about attention performance can then be answered by reading a
    log line instead of asking the user to run a probe script. Call it from a generation path rather
    than from startup -- `probe_attention_backends` explains why.
    """
    key = str(device)
    if key in _probed_attention_devices:
        return
    _probed_attention_devices.add(key)
    available = probe_attention_backends(device)
    if available is None:
        return
    summary = " ".join(f"{name}={'yes' if ok else 'no'}" for name, ok in available.items())
    logger.info(
        f"SDPA attention backends on {device} "
        f"(fp16, {_PROBE_HEADS} heads, seq {_PROBE_SEQ_LEN}, head_dim {_PROBE_HEAD_DIM}, no mask): {summary}; "
        f"priority {describe_sdpa_priority_order()}"
    )


def invokeai_source_dir() -> Path:
    # `invokeai.__file__` doesn't always work for editable installs
    this_module_path = Path(__file__).resolve()
    # https://youtrack.jetbrains.com/issue/PY-38382/Unresolved-reference-spec-but-this-is-standard-builtin
    # noinspection PyUnresolvedReferences
    depth = len(__spec__.parent.split("."))
    return this_module_path.parents[depth - 1]


def enable_dev_reload(custom_nodes_path=None) -> None:
    """Enable hot reloading on python file changes during development."""
    from invokeai.backend.util.logging import InvokeAILogger

    try:
        import jurigged
    except ImportError as e:
        raise RuntimeError(
            'Can\'t start `--dev_reload` because jurigged is not found; `pip install -e ".[dev]"` to include development dependencies.'
        ) from e
    else:
        paths = [str(invokeai_source_dir() / "*.py")]
        if custom_nodes_path:
            paths.append(str(custom_nodes_path / "*.py"))
        jurigged.watch(pattern=paths, logger=InvokeAILogger.get_logger(name="jurigged").info)


def apply_monkeypatches(attention_backend: ATTENTION_BACKEND = "auto") -> None:
    """Apply monkeypatches to fix issues with third-party libraries, and install the configured attention backend."""

    import invokeai.backend.util.hotfixes  # noqa: F401 (monkeypatching on import)

    # Third-party libraries (diffusers calls empty_device_cache inside every from_pretrained /
    # from_single_file) invoke torch.cuda.empty_cache directly, convoying a busy peer GPU on
    # multi-GPU installs. Wrap the torch entry point so every Python caller inherits the
    # skip-while-a-peer-generates policy.
    from invokeai.backend.util.devices import install_cuda_stream_capture_shim, install_peer_aware_empty_cache

    install_peer_aware_empty_cache()
    # NVIDIA's Windows ARM64 torch crashes in torch.cuda.is_current_stream_capturing() without a device.
    install_cuda_stream_capture_shim()

    # ROCm's fused SDPA kernels return wrong output for heads wider than 256 (the VAE mid-block
    # attention), turning decodes into noise or black images. Route those calls to the math kernel,
    # and compute math-kernel attention -- the only kernel ROCm on Windows has -- in bounded chunks.
    from invokeai.backend.util.attention import install_rocm_sdpa_guard

    install_rocm_sdpa_guard()

    # Rank torch's exact SDPA kernels flash, cuDNN, memory-efficient, math for every model in the process, with a
    # pinned kernel ahead of them. The calls SageAttention does not serve use this order too.
    pinned = SDPA_KERNELS.get(attention_backend)
    if install_default_sdpa_priority(pinned) and pinned is not None:
        # A pinned kernel is never exclusive: log the attention it does not serve. Outside the ROCm guard, so it sees
        # the heads the guard sends to math; inside the cuDNN fallback, so a retry is reported on its kernel.
        from invokeai.backend.util.attention import install_pinned_sdpa_kernel_report

        install_pinned_sdpa_kernel_report(pinned)

    # cuDNN's eligibility is checked statically, so a cuDNN attention call can still fail at run time. Demote cuDNN
    # and retry instead of failing the generation. Before SageAttention, so its own fallback calls are covered too.
    from invokeai.backend.util.attention import install_cudnn_sdpa_fallback

    install_cudnn_sdpa_fallback()

    # Opt-in SageAttention for diffusion-model attention. Installs nothing unless asked for, and never on ROCm.
    if attention_backend == "sage":
        from invokeai.backend.util.sage_attention import install_sage_attention

        install_sage_attention()


def register_mime_types() -> None:
    """Register additional mime types for windows."""
    # Fix for windows mimetypes registry entries being borked.
    # see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
