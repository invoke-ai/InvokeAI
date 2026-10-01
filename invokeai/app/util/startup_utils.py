import logging
import mimetypes
import socket
from pathlib import Path

import torch


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


# Devices already reported. The line answers a support question ("which kernels does this build
# have?"), so it is worth exactly once per device per process and nothing after that.
_logged_attention_backends: set[str] = set()


def log_attention_backends(logger: logging.Logger, device: torch.device) -> None:
    """Log this device's SDPA backend availability, once per device.

    The point is support: a question about attention performance can then be answered by reading a
    log line instead of asking the user to run a probe script. Call it from a generation path rather
    than from startup -- `probe_attention_backends` explains why.
    """
    key = str(device)
    if key in _logged_attention_backends:
        return
    available = probe_attention_backends(device)
    if available is None:
        return
    _logged_attention_backends.add(key)
    summary = " ".join(f"{name}={'yes' if ok else 'no'}" for name, ok in available.items())
    logger.info(
        f"SDPA attention backends on {device} "
        f"(fp16, {_PROBE_HEADS} heads, seq {_PROBE_SEQ_LEN}, head_dim {_PROBE_HEAD_DIM}, no mask): {summary}"
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


def apply_monkeypatches() -> None:
    """Apply monkeypatches to fix issues with third-party libraries."""

    import invokeai.backend.util.hotfixes  # noqa: F401 (monkeypatching on import)

    # Third-party libraries (diffusers calls empty_device_cache inside every from_pretrained /
    # from_single_file) invoke torch.cuda.empty_cache directly, convoying a busy peer GPU on
    # multi-GPU installs. Wrap the torch entry point so every Python caller inherits the
    # skip-while-a-peer-generates policy.
    from invokeai.backend.util.devices import install_peer_aware_empty_cache

    install_peer_aware_empty_cache()

    # ROCm's fused SDPA kernels return wrong output for heads wider than 256 (the VAE mid-block
    # attention), turning decodes into noise or black images. Route those calls to the math kernel,
    # and compute math-kernel attention -- the only kernel ROCm on Windows has -- in bounded chunks.
    from invokeai.backend.util.attention import install_rocm_sdpa_guard

    install_rocm_sdpa_guard()


def register_mime_types() -> None:
    """Register additional mime types for windows."""
    # Fix for windows mimetypes registry entries being borked.
    # see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
