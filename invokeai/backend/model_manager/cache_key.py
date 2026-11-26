"""Model cache identity and measurement ownership, without importing the cache runtime."""

import re
from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager.taxonomy import ModelType, SubModelType

_SUBMODEL_TYPES = frozenset(submodel.value for submodel in SubModelType)
_ROLE_SUFFIXES = frozenset(f"as_{model_type.value}" for model_type in ModelType)
_STORAGE_VARIANT = re.compile(r"(?:fp8-|full-)?(?:float16|bfloat16|float32)(?:-layerwise)?")
_LEGACY_T5_SUBMODELS = frozenset({SubModelType.TextEncoder2.value, SubModelType.TextEncoder3.value})


def get_model_cache_key(
    model_key: str,
    submodel_type: Optional[SubModelType] = None,
    *,
    role: Optional[ModelType] = None,
    variant: str = "",
) -> str:
    key = f"{model_key}:{submodel_type.value}" if submodel_type else model_key
    if role is not None:
        key = f"{key}:as_{role.value}"
    return f"{key}:storage={variant}" if variant else key


def get_model_part_cache_key(model_key: str, part: str, path: Path) -> str:
    """The key of a part of an installed model that is read from the model's file by its own loader.

    The `<model_key>:` prefix makes `drop_model` drop it with the model. The `part:` infix keeps it apart from the
    loader's submodel, role and storage entries. The path keeps a record repointed to another file from reusing it.
    """
    return f"{model_key}:part:{part}:{path}"


def model_key_from_cache_key(cache_key: str) -> str:
    """Remove only recognized cache suffixes; model keys themselves may contain colons.

    Keep the legacy T5 decoder. A bare role suffix or an unmarked Qwen dtype can be part of an exact
    historical model ID. Decode roles only with a storage marker and leave unmarked Qwen keys intact.
    """
    head, separator, tail = cache_key.rpartition(":")
    if separator and tail.startswith("storage=") and _STORAGE_VARIANT.fullmatch(tail.removeprefix("storage=")):
        cache_key = head
        model_and_submodel, role_separator, role = head.rpartition(":")
        if role_separator and role in _ROLE_SUFFIXES:
            cache_key = model_and_submodel
    elif separator and _STORAGE_VARIANT.fullmatch(tail):
        _owner, slot_separator, slot = head.rpartition(":")
        if slot_separator and slot in _LEGACY_T5_SUBMODELS:
            cache_key = head
    head, separator, tail = cache_key.rpartition(":")
    return head if separator and tail in _SUBMODEL_TYPES else cache_key
