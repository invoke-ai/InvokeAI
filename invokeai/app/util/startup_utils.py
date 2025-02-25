import logging
import mimetypes
import socket

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


def enable_dev_reload() -> None:
    """Enable hot reloading on python file changes during development."""
    from invokeai.backend.util.logging import InvokeAILogger

    try:
        import jurigged
    except ImportError as e:
        raise RuntimeError(
            'Can\'t start `--dev_reload` because jurigged is not found; `pip install -e ".[dev]"` to include development dependencies.'
        ) from e
    else:
        jurigged.watch(logger=InvokeAILogger.get_logger(name="jurigged").info)


def apply_monkeypatches() -> None:
    """Apply monkeypatches to fix issues with third-party libraries."""

    import invokeai.backend.util.hotfixes  # noqa: F401 (monkeypatching on import)

    if torch.backends.mps.is_available():
        import invokeai.backend.util.mps_fixes  # noqa: F401 (monkeypatching on import)


def register_mime_types() -> None:
    """Register additional mime types for windows."""
    # Fix for windows mimetypes registry entries being borked.
    # see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
