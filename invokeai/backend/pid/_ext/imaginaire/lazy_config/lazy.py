# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal LazyCall / LazyConfig stub. The upstream module supports file-based
# config save/load via yaml + cloudpickle + dill + detectron2 helpers; the
# vendored decoder-inference subset only needs `LazyCall(cls)(**kwargs)` as a
# convenient producer of `{_target_: "cls.fqn", **kwargs}` dicts that
# `instantiate()` can resolve.

from typing import Any

from invokeai.backend.pid._ext.imaginaire.lazy_config.registry import _convert_target_to_string

__all__ = ["LazyCall", "LazyConfig"]


class _LazyCallResult(dict):
    """A plain dict tagged for `instantiate()`. Behaves like a DictConfig
    enough for our subset (attribute access falls back to item access)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class LazyCall:
    """`LazyCall(cls)(**kwargs)` -> `{_target_: <fqn>, **kwargs}`."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __call__(self, **kwargs: Any) -> _LazyCallResult:
        target_str = _convert_target_to_string(self._target) if not isinstance(self._target, str) else self._target
        return _LazyCallResult(_target_=target_str, **kwargs)


class LazyConfig:
    """File-IO helpers from the upstream module are not used in the inference
    subset and are intentionally omitted."""

    @staticmethod
    def load(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("LazyConfig.load is not supported in the vendored PiD inference subset.")

    @staticmethod
    def save(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("LazyConfig.save is not supported in the vendored PiD inference subset.")
