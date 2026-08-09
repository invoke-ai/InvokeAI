# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal stdlib-based stand-in for the upstream iopath PathManager.
# Only `open()` on local paths and trivial helpers are supported; the upstream
# HTTPURLHandler / OneDrivePathHandler paths are not used by the decoder
# inference subset we vendor.

import io
import shutil
from typing import IO, Any

__all__ = ["PathManager", "PathHandler"]


class PathHandler:
    """Base no-op handler (kept for API parity)."""

    def _open(self, path: str, mode: str = "r", **kwargs: Any) -> IO:
        return io.open(path, mode, **kwargs)


class _LocalPathManager:
    def open(self, path: str, mode: str = "r", **kwargs: Any) -> IO:
        return io.open(path, mode, **kwargs)

    def get_local_path(self, path: str, **kwargs: Any) -> str:
        return path

    def exists(self, path: str) -> bool:
        import os.path

        return os.path.exists(path)

    def isfile(self, path: str) -> bool:
        import os.path

        return os.path.isfile(path)

    def isdir(self, path: str) -> bool:
        import os.path

        return os.path.isdir(path)

    def mkdirs(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)

    def copy(self, src: str, dst: str, overwrite: bool = False) -> bool:
        shutil.copy(src, dst)
        return True

    def register_handler(self, handler: PathHandler) -> None:
        pass


PathManager = _LocalPathManager()
