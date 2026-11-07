"""Model cache identity and measurement ownership, without importing the cache runtime."""

import re
from typing import Optional

from invokeai.backend.model_manager.taxonomy import SubModelType

_SUBMODEL_TYPES = frozenset(submodel.value for submodel in SubModelType)
_STORAGE_VARIANT = re.compile(r"(?:fp8-|full-)?(?:float16|bfloat16|float32)(?:-layerwise)?")
_LEGACY_T5_SUBMODELS = frozenset({SubModelType.TextEncoder2.value, SubModelType.TextEncoder3.value})


def get_model_cache_key(model_key: str, submodel_type: Optional[SubModelType] = None, *, variant: str = "") -> str:
    key = f"{model_key}:{submodel_type.value}" if submodel_type else model_key
    return f"{key}:storage={variant}" if variant else key


def model_key_from_cache_key(cache_key: str) -> str:
    """Remove only recognized cache suffixes; model keys themselves may contain colons.

    T5 keys predating the explicit storage marker have a dtype after text_encoder_2/3. Unknown
    tails retain their historical ownership rather than guessing that their prefix is a model.
    """
    head, separator, tail = cache_key.rpartition(":")
    if separator and tail.startswith("storage=") and _STORAGE_VARIANT.fullmatch(tail.removeprefix("storage=")):
        cache_key = head
    elif separator and _STORAGE_VARIANT.fullmatch(tail):
        _owner, slot_separator, slot = head.rpartition(":")
        if slot_separator and slot in _LEGACY_T5_SUBMODELS:
            cache_key = head
    head, separator, tail = cache_key.rpartition(":")
    return head if separator and tail in _SUBMODEL_TYPES else cache_key
