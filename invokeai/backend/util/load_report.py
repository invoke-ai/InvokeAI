"""Suppression of the transformers state-dict load report where it reports only expected keys."""

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

# transformers emits the report from `modeling_utils`, at warning level, through this logger.
_REPORT_LOGGER_NAME = "transformers.modeling_utils"
# Its first line is "<ClassName> LOAD REPORT from: <path>" (the class name ANSI-styled),
# followed by one table of every key that did not line up.
_REPORT_MARKER = "LOAD REPORT"
# The table's "Notes" block explains one category per kind of row it contains. UNEXPECTED is
# the only one a tower load is entitled to; the others mean weights were randomly initialized,
# reinitialized at a different shape, or failed to convert — i.e. a checkpoint that is not what
# the class being built expects, which is exactly what must stay in the log. The markers carry
# the colon only in that block, so a key named like a status cannot be mistaken for one.
_DIAGNOSTIC_MARKERS = ("MISSING:", "MISMATCH:", "CONVERSION:")

_state = threading.local()


class _LoadReportFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(_state, "active", False):
            return True
        message = record.getMessage()
        if _REPORT_MARKER not in message:
            return True
        return any(marker in message for marker in _DIAGNOSTIC_MARKERS)


# Installed once, on the emitting logger rather than on a handler: transformers keeps its own
# handler and sets `propagate = False` on its root logger, so a filter installed anywhere on
# InvokeAI's side would never see these records.
logging.getLogger(_REPORT_LOGGER_NAME).addFilter(_LoadReportFilter())


@contextmanager
def suppress_load_report() -> Iterator[None]:
    """Drop transformers' all-expected state-dict load report for a load made on this thread.

    Building a vision or text tower out of a full CLIP/SigLIP checkpoint always mismatches —
    every key of the tower that is not being built is unexpected, by design — so the report is
    a two-dozen-line table of keys that are supposed to be there, logged as a warning, on a
    load that worked. A report that also names missing, mismatched or unconvertible keys is
    kept: those say the checkpoint is not the one the class expects (a tower silently built
    with randomly initialized weights embeds and conditions on noise), and transformers refers
    to the report from the errors it raises for the latter two.

    Scoped to the calling thread rather than lowering `transformers` verbosity, because that is
    process-global: model loads run concurrently with generation and with the image index
    worker, so a verbosity toggle here would silence another thread's warnings, and its restore
    would un-silence a load still in flight. Every other transformers message, including one
    from the load inside this block, still reaches the log.
    """
    previously_active = getattr(_state, "active", False)
    _state.active = True
    try:
        yield
    finally:
        _state.active = previously_active
