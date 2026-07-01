# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# stdlib-based replacement for the upstream loguru-based logger.
# Provides a drop-in `logger` plus `info/warning/error/...` module-level
# functions so vendored call sites do not need to be touched.

import logging
from typing import Any

logger = logging.getLogger("invokeai.backend.pid")


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.info(str(msg), *args)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.warning(str(msg), *args)


warn = warning


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.error(str(msg), *args)


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.debug(str(msg), *args)


def critical(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.critical(str(msg), *args)


def exception(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.exception(str(msg), *args)


def trace(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.debug(str(msg), *args)


def success(msg: Any, *args: Any, **kwargs: Any) -> None:
    logger.info(str(msg), *args)


def init_loguru_stdout() -> None:
    pass


def init_loguru_file(path: str) -> None:
    pass
