import pytest

from invokeai.backend.model_management.libc_util import LibcUtil


def test_libc_util_mallinfo2():
    """Smoke test of LibcUtil().mallinfo2()."""
    try:
        libc = LibcUtil()
    except OSError:
        # TODO: Set the expected result preemptively based on the system properties.
        pytest.xfail("libc shared library is not available on this system.")

    info = libc.mallinfo2()

    assert info.arena > 0
