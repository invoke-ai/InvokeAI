# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pydoc
from typing import Any


class Registry:
    """Minimal stand-in for fvcore.common.registry.Registry.

    Only the subset used by the vendored PiD decode path is implemented:
    name-keyed object registry with ``register``/``get``.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._obj_map: dict[str, Any] = {}

    def register(self, obj: Any = None, *, name: str | None = None) -> Any:
        if obj is None:

            def deco(x: Any) -> Any:
                self._do_register(name or x.__name__, x)
                return x

            return deco
        self._do_register(name or obj.__name__, obj)
        return obj

    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            raise KeyError(f"{name} already registered in {self._name}")
        self._obj_map[name] = obj

    def get(self, name: str) -> Any:
        if name not in self._obj_map:
            raise KeyError(f"{name} not found in {self._name}")
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())


"""
``Registry`` and `locate` provide ways to map a string (typically found
in config files) to callable objects.
"""

__all__ = ["Registry", "locate"]


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    obj = pydoc.locate(name)
    if obj is None:
        # Fallback: walk the module path manually for cases pydoc.locate misses
        # (e.g. nested classes, re-exports).
        import importlib

        parts = name.split(".")
        for k in range(len(parts) - 1, 0, -1):
            mod_path, attr_path = ".".join(parts[:k]), parts[k:]
            try:
                obj = importlib.import_module(mod_path)
                for a in attr_path:
                    obj = getattr(obj, a)
                break
            except (ImportError, AttributeError):
                obj = None
        if obj is None:
            raise ImportError(f"Cannot dynamically locate object {name}!")
    return obj
