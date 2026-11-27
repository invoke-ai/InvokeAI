"""`silence_int8_cast_notice`, which must work without bitsandbytes installed (macOS has none)."""

import logging
import warnings

import torch

from invokeai.backend.quantization.bnb_cast_notice import silence_int8_cast_notice

_CAST_NOTICE = "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"


def test_int8_cast_notice_is_silenced_as_a_warning_and_as_a_log_record(caplog):
    """bitsandbytes up to 0.49 warns with `warnings`; from 0.50 it logs. Other notices from either path still show."""
    bnb_logger = logging.getLogger("bitsandbytes.autograd._functions")
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING, logger=bnb_logger.name):
        warnings.simplefilter("always")
        silence_int8_cast_notice()
        silence_int8_cast_notice()  # idempotent: one log filter, not two

        warnings.warn(_CAST_NOTICE, UserWarning, stacklevel=1)
        warnings.warn("an unrelated bitsandbytes warning", UserWarning, stacklevel=1)
        bnb_logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", torch.bfloat16)
        bnb_logger.warning("an unrelated bitsandbytes log record")

    assert [str(w.message) for w in caught] == ["an unrelated bitsandbytes warning"]
    assert [r.getMessage() for r in caplog.records] == ["an unrelated bitsandbytes log record"]
    assert sum(type(f).__name__ == "_DropInt8CastNotice" for f in bnb_logger.filters) == 1
