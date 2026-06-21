# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Upstream re-exported `LazyDict = omegaconf.DictConfig`; in this vendored
# subset configs are plain Python mappings, so `LazyDict` aliases the
# attribute-accessible dict subclass produced by `LazyCall`.

from invokeai.backend.pid._ext.imaginaire.lazy_config.instantiate import instantiate
from invokeai.backend.pid._ext.imaginaire.lazy_config.lazy import LazyCall, LazyConfig, _LazyCallResult

PLACEHOLDER = None
LazyDict = _LazyCallResult

__all__ = ["instantiate", "LazyCall", "LazyConfig", "PLACEHOLDER", "LazyDict"]
