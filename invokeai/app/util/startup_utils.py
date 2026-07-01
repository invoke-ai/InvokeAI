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


def register_mime_types() -> None:
    """Register additional mime types for windows."""
    # Fix for windows mimetypes registry entries being borked.
    # see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
