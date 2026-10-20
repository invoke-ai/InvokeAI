"""Why bitsandbytes could not be loaded, for the guards that carry on without it.

Several modules import bitsandbytes optionally and degrade when it fails. bitsandbytes may be absent (macOS) or
installed and fail to load its native library; only the second is a fault worth reporting, and the error that
says why is gone by the time a load needs bitsandbytes. Kept apart from the bnb modules so it imports without
bitsandbytes.
"""

from typing import Optional

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

_failure: Optional[BaseException] = None


def _is_absent(error: BaseException) -> bool:
    return isinstance(error, ModuleNotFoundError) and (error.name or "").partition(".")[0] == "bitsandbytes"


def record_bnb_import_failure(error: BaseException) -> None:
    """Keep the first failure, and log it once unless bitsandbytes is simply not installed."""
    global _failure
    if _failure is not None:
        return
    _failure = error
    if not _is_absent(error):
        logger.warning(
            f"bitsandbytes is installed but failed to load; bitsandbytes-quantized models will not load: {error!r}"
        )


def bnb_unavailable_message() -> str:
    """The error text for a load that needs bitsandbytes when it is unavailable."""
    if _failure is None or _is_absent(_failure):
        return "The bnb modules are not available. Please install bitsandbytes if available on your platform."
    return f"bitsandbytes is installed but failed to load: {_failure!r}"
