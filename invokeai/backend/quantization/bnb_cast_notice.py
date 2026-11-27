"""Silence bitsandbytes' per-matmul notice that LLM.int8 casts bf16 activations to fp16.

Kept apart from `bnb_llm_int8` so it imports without bitsandbytes, which is not installed on macOS: callers that load
an int8 model through transformers install the filter before bitsandbytes is ever imported.
"""

import logging
import warnings

_INT8_CAST_NOTICE = "MatMul8bitLt: inputs will be cast from"


class _DropInt8CastNotice(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_INT8_CAST_NOTICE)


def silence_int8_cast_notice() -> None:
    """Silence bitsandbytes' notice that LLM.int8 casts bf16 activations to fp16.

    The int8 matmul kernel only takes fp16 activations, so for our bf16 encoders the cast is correct and intended, but
    bitsandbytes reports it on *every* matmul of *every* layer: as a UserWarning up to 0.49, through the
    `bitsandbytes.autograd._functions` logger from 0.50, which `warnings` filters do not reach. Idempotent.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
        category=UserWarning,
    )
    bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
    if not any(isinstance(existing, _DropInt8CastNotice) for existing in bnb_logger.filters):
        bnb_logger.addFilter(_DropInt8CastNotice())
