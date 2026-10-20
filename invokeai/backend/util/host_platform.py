import platform
import sys


def is_windows_arm64() -> bool:
    """Whether this is Windows on ARM64, where several dependencies have no wheel and are left out of the install.

    Read at call time, not import time, so tests can patch ``sys.platform`` and ``platform.machine``.
    """
    return sys.platform == "win32" and platform.machine() == "ARM64"
