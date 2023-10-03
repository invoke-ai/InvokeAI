import pytest

from invokeai.backend.model_management.libc_util import LibcUtil, Struct_mallinfo2


def test_libc_util_mallinfo2():
    """Smoke test of LibcUtil().mallinfo2()."""
    try:
        libc = LibcUtil()
    except OSError:
        # TODO: Set the expected result preemptively based on the system properties.
        pytest.xfail("libc shared library is not available on this system.")

    info = libc.mallinfo2()

    assert info.arena > 0


def test_struct_mallinfo2_to_str():
    """Smoke test of Struct_mallinfo2.__str__()."""
    info = Struct_mallinfo2()
    info_str = str(info)
    print(info_str)
