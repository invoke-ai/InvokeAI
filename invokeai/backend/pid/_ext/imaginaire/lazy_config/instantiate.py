# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stdlib-only `instantiate()`. The upstream module also handled
# omegaconf.DictConfig / ListConfig structured configs and OmegaConf.to_object
# round-trips. In the vendored decoder-inference subset all configs are
# constructed as plain Python mappings (see invokeai/backend/pid/decode.py),
# so the omegaconf paths are not required.

import collections.abc as abc
import dataclasses
import logging
from typing import Any

import attrs

from invokeai.backend.pid._ext.imaginaire.lazy_config.registry import _convert_target_to_string, locate

__all__ = ["dump_dataclass", "instantiate"]


def is_dataclass_or_attrs(target: Any) -> bool:
    return dataclasses.is_dataclass(target) or attrs.has(target)


def dump_dataclass(obj: Any) -> dict:
    """Recursively dump a dataclass into a dict that can be re-instantiated."""
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type), (
        "dump_dataclass() requires an instance of a dataclass."
    )
    ret: dict = {"_target_": _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]
        ret[f.name] = v
    return ret


def instantiate(cfg: Any, *args: Any, **kwargs: Any) -> Any:
    """Recursively instantiate objects defined by `_target_` + arguments.

    Accepts any Mapping with a `_target_` key (e.g. plain dict or the
    `_LazyCallResult` produced by `LazyCall`). Lists are walked recursively.
    """
    if isinstance(cfg, list):
        return [instantiate(x) for x in cfg]

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        is_recursive = bool(cfg.get("_recursive_", True))
        if is_recursive:
            resolved = {k: instantiate(v) for k, v in cfg.items()}
        else:
            resolved = dict(cfg)
        resolved.pop("_recursive_", None)
        cls = resolved.pop("_target_")
        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            cls_name = getattr(cls, "__qualname__", str(cls))
        assert callable(cls), f"_target_ {cls_name} does not define a callable object"
        try:
            return cls(*args, **{**resolved, **kwargs})
        except TypeError:
            logging.getLogger(__name__).error("Error when instantiating %s!", cls_name)
            raise

    return cfg
